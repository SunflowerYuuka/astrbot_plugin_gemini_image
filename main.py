"""
Gemini Image Generation Plugin
使用 Gemini 系列模型进行图像生成的插件
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Coroutine
from typing import Any

import aiohttp
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config.astrbot_config import AstrBotConfig

from .gemini_generator import GeminiImageGenerator


@pydantic_dataclass
class GeminiImageGenerationTool(FunctionTool[AstrAgentContext]):
    """统一的图像生成工具，支持文生图和图生图"""

    name: str = "gemini_generate_image"
    description: str = "使用 Gemini 模型生成图片。"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生成图片的详细描述,需要是英文提示词。如果是文生图,描述风格、主题、细节等信息;如果是图生图,描述想要如何修改或变换图片",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比。可选值: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9。默认: 1:1",
                    "enum": [
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                },
                "resolution": {
                    "type": "string",
                    "description": "图片分辨率(仅 gemini-3-pro-image-preview 模型支持)。可选值: 1K(标准分辨率), 2K(高分辨率), 4K(超高分辨率)。默认: 1K。注意: 更高分辨率需要更多生成时间",
                    "enum": ["1K", "2K", "4K"],
                },
                "use_reference_image": {
                    "type": "boolean",
                    "description": "是否使用参考图片。true 表示图生图模式(基于用户发送的图片),false 表示文生图模式(纯文字生成)。默认: false",
                },
                "reference_image_index": {
                    "type": "number",
                    "description": "参考图片的索引,从0开始。仅在 use_reference_image=true 时有效。默认使用最新的图片(0)",
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        prompt = kwargs.get("prompt", "")
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")
        resolution = kwargs.get("resolution", "1K")
        use_reference_image = kwargs.get("use_reference_image", False)
        image_index = int(kwargs.get("reference_image_index", 0))

        if not prompt:
            return "请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            try:
                plugin = context.context.context
            except Exception:
                plugin = None

        if not plugin:
            return "插件未正确初始化"

        # 获取事件
        event = None
        try:
            event = context.context.event
        except Exception:
            pass

        if not event:
            return "无法获取当前消息事件"

        # 根据参数决定是否使用参考图片
        image_data = None
        mime_type = None

        if use_reference_image:
            recent_images = plugin.get_recent_images(event.unified_msg_origin)
            if not recent_images or image_index >= len(recent_images):
                return f"未找到参考图片。请让用户先发送图片。当前可用图片数: {len(recent_images) if recent_images else 0}"

            ref_image = recent_images[image_index]
            image_data = ref_image["data"]
            mime_type = ref_image["mime_type"]

        # 创建异步任务,在后台生成图片
        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                image_data=image_data,
                mime_type=mime_type,
                unified_msg_origin=event.unified_msg_origin,
                use_reference_image=use_reference_image,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
        )

        # 返回空字符串,让 LLM 自己决定如何回应,避免提前说"完成"
        return ""


class GeminiImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()

        # 读取配置
        self._load_config()

        # 初始化生成器
        self.generator = GeminiImageGenerator(
            api_keys=self.api_keys,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            cache_ttl=self.cache_ttl,
            max_cache_count=self.max_cache_count,
        )

        # 存储最近收到的图片 {session_id: [{"data": bytes, "mime_type": str, "timestamp": float}]}
        self.recent_images: dict[str, list[dict]] = {}
        self.max_images_per_session = 5

        # 异步任务追踪
        self.background_tasks: set[asyncio.Task] = set()

        # 注册工具到 LLM
        if self.enable_llm_tool:
            # 将插件实例注入到工具中，方便工具在执行时访问生成器和缓存
            self.context.add_llm_tools(GeminiImageGenerationTool(plugin=self))
            logger.info("[Gemini Image] 已注册统一的图像生成工具")

        logger.info(f"[Gemini Image] 插件已加载，使用模型: {self.model}")

    def _load_config(self):
        """加载配置"""
        # 从系统提供商中获取配置
        use_system_provider = self.config.get("use_system_provider", True)
        provider_id = (self.config.get("provider_id", "") or "").strip()

        if use_system_provider and provider_id:
            # 使用系统提供商配置
            provider = self.context.get_provider_by_id(provider_id)
            if provider:
                api_keys, api_base = self._extract_provider_credentials(provider)
                if api_keys:
                    self.api_keys = api_keys
                    self.base_url = (
                        api_base or "https://generativelanguage.googleapis.com"
                    )
                    logger.info(
                        f"[Gemini Image] 使用系统提供商: {provider_id}，API Keys 数量: {len(self.api_keys)}"
                    )
                else:
                    logger.warning(
                        f"[Gemini Image] 提供商 {provider_id} 未提供可用的 API Key，将使用插件配置"
                    )
                    self._load_default_config()
            else:
                logger.warning(
                    f"[Gemini Image] 未找到提供商 {provider_id}，将使用插件配置"
                )
                self._load_default_config()
        else:
            # 使用插件默认配置
            if use_system_provider and not provider_id:
                logger.warning("[Gemini Image] 未配置提供商 ID，将使用插件配置")
            self._load_default_config()

        # 加载其他配置
        self.model = self.config.get("model", "gemini-2.0-flash-exp-image-generation")
        # 如果选择了自定义模型，使用 custom_model 配置
        if self.model == "custom":
            custom_model = self.config.get("custom_model", "").strip()
            if custom_model:
                self.model = custom_model
                logger.info(f"[Gemini Image] 使用自定义模型: {self.model}")
            else:
                logger.warning(
                    "[Gemini Image] 选择了 custom 但未配置 custom_model，将使用默认模型"
                )
                self.model = "gemini-2.0-flash-exp-image-generation"

        self.timeout = self.config.get("timeout", 120)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.max_cache_count = self.config.get("max_cache_count", 100)
        self.enable_llm_tool = self.config.get("enable_llm_tool", True)

    def _load_default_config(self):
        """加载默认配置"""
        api_key = self.config.get("api_key", "")
        # 支持单个key或多个key
        if isinstance(api_key, list):
            self.api_keys = [k for k in api_key if k]
        elif isinstance(api_key, str) and api_key:
            self.api_keys = [api_key]
        else:
            self.api_keys = []

        self.base_url = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com"
        )
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

    def _extract_provider_credentials(
        self,
        provider: object,
    ) -> tuple[list[str], str | None]:
        """从 Provider 实例提取 API Keys 与 Base URL

        Returns:
            (API Keys 列表, Base URL)
        """

        api_keys = []
        api_base: str | None = None

        provider_config = getattr(provider, "provider_config", {}) or {}
        keys = provider_config.get("key") or provider_config.get("keys")

        # 支持多种 key 配置格式
        if isinstance(keys, str):
            if keys:
                api_keys = [keys]
        elif isinstance(keys, list):
            api_keys = [k for k in keys if k]
        else:
            # 尝试其他可能的 key 字段
            extra_key = provider_config.get("api_key") or provider_config.get(
                "access_token",
            )
            if isinstance(extra_key, str) and extra_key:
                api_keys = [extra_key]
            elif isinstance(extra_key, list):
                api_keys = [k for k in extra_key if k]

        api_base = (
            getattr(provider, "api_base", None)
            or provider_config.get("api_base")
            or provider_config.get("api_base_url")
        )
        if isinstance(api_base, str) and api_base.endswith("/"):
            api_base = api_base.rstrip("/")

        return api_keys, api_base

    @filter.command("img")
    async def generate_image_command(self, event: AstrMessageEvent):
        """生成图片指令

        用法:
        /img <提示词> - 文生图
        /img <提示词> (引用包含图片的消息) - 图生图
        """
        prompt = event.message_str.strip()
        if not prompt:
            yield event.plain_result("请提供生成图片的提示词！\n用法: /img <提示词>")
            return

        # 检查是否有引用的图片或消息中的图片
        has_image = False
        image_data = None
        mime_type = "image/jpeg"

        # 从消息链中查找图片
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                has_image = True
                # 下载图片
                try:
                    image_url = component.url or component.file
                    if image_url:
                        import aiohttp

                        async with aiohttp.ClientSession() as session:
                            async with session.get(image_url) as resp:
                                if resp.status == 200:
                                    image_data = await resp.read()
                                    content_type = resp.headers.get(
                                        "Content-Type", "image/jpeg"
                                    )
                                    mime_type = content_type
                                    logger.info(
                                        f"[Gemini Image] 下载图片成功: {len(image_data)} bytes"
                                    )
                                else:
                                    logger.error(
                                        f"[Gemini Image] 下载图片失败: {resp.status}"
                                    )
                except Exception as e:
                    logger.error(f"[Gemini Image] 下载图片时出错: {e}")

                break

        # 尝试从消息中抓取第一张图片
        if not image_data:
            image_result = await self._first_image_from_event(event)
            if image_result:
                image_data, mime_type = image_result
                has_image = True

        # 注意: /genimg 命令不使用缓存的图片,只有在消息中明确包含图片时才使用图生图

        # 立即响应用户
        mode = "图生图" if (has_image and image_data) else "文生图"
        yield event.plain_result(f"已开始{mode}任务，图片生成需要 10-30 秒，请稍候...")

        # 创建异步任务,在后台生成图片
        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                image_data=image_data if (has_image and image_data) else None,
                mime_type=mime_type if (has_image and image_data) else None,
                unified_msg_origin=event.unified_msg_origin,
                use_reference_image=(has_image and image_data is not None),
            )
        )

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听消息，缓存用户发送的图片"""
        # 从消息中提取图片
        for component in event.message_obj.message:
            if not isinstance(component, Comp.Image):
                continue

            image_url = component.url or component.file
            result = await self._download_image(image_url)
            if not result:
                continue

            image_data, mime_type = result
            self._remember_user_image(event.unified_msg_origin, image_data, mime_type)

    def get_recent_images(self, session_id: str) -> list[dict]:
        """获取会话的最近图片"""
        return self.recent_images.get(session_id, [])

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """统一创建后台任务并追踪生命周期"""

        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    async def _download_image(self, image_url: str | None) -> tuple[bytes, str] | None:
        """下载图片并返回数据与 MIME 类型"""

        if not image_url:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        logger.error(
                            f"[Gemini Image] 下载图片失败: {resp.status} - {image_url}"
                        )
                        return None

                    image_data = await resp.read()
                    mime_type = resp.headers.get("Content-Type", "image/jpeg")
                    logger.info(
                        f"[Gemini Image] 下载图片成功: {len(image_data)} bytes"  # noqa: G004
                    )
                    return image_data, mime_type
        except Exception as exc:
            logger.error(f"[Gemini Image] 下载图片时出错: {exc}")
            return None

    async def _first_image_from_event(
        self, event: AstrMessageEvent
    ) -> tuple[bytes, str] | None:
        """获取消息链中的第一张图片"""

        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                result = await self._download_image(component.url or component.file)
                if result:
                    return result
        return None

    def _remember_user_image(
        self, session_id: str, image_data: bytes, mime_type: str | None
    ) -> None:
        """缓存用户发送的图片以便作为参考"""

        mime = mime_type or "image/jpeg"
        session_images = self.recent_images.setdefault(session_id, [])
        session_images.insert(
            0,
            {
                "data": image_data,
                "mime_type": mime,
                "timestamp": time.time(),
            },
        )

        if len(session_images) > self.max_images_per_session:
            del session_images[self.max_images_per_session :]

        logger.info(
            f"[Gemini Image] 已缓存用户图片，会话 {session_id} 当前有 {len(session_images)} 张图片"
        )

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        image_data: bytes | None = None,
        mime_type: str | None = None,
        use_reference_image: bool = False,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
    ):
        """异步生成图片并发送给用户"""
        try:
            logger.info(
                f"[Gemini Image] 开始异步生成任务，会话: {unified_msg_origin}，提示词: {prompt[:50]}..."
            )

            # 调用统一的生成接口
            result_data, error = await self.generator.generate_image(
                prompt=prompt,
                image_data=image_data,
                mime_type=mime_type,
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            )

            if error:
                # 发送错误消息
                error_msg = f"图片生成失败: {error}"
                logger.error(f"[Gemini Image] {error_msg}")
                await self.context.send_message(
                    unified_msg_origin,
                    MessageChain().text(error_msg),
                )
                return

            # 缓存图片
            image_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()
            file_path = await self.generator.cache_image(image_id, result_data)

            # 发送图片给用户
            message_chain = MessageChain().file_image(str(file_path))
            await self.context.send_message(unified_msg_origin, message_chain)

            mode = "图生图" if use_reference_image else "文生图"
            logger.info(f"[Gemini Image] {mode}任务完成，已发送给用户")

        except Exception as e:
            logger.error(f"[Gemini Image] 异步生成任务失败: {e}", exc_info=True)
            try:
                await self.context.send_message(
                    unified_msg_origin,
                    MessageChain().text(f"图片生成过程中发生错误: {str(e)}"),
                )
            except Exception:
                pass

    async def terminate(self):
        """插件卸载时清理资源"""
        try:
            # 取消所有后台任务
            if hasattr(self, "background_tasks"):
                pending_count = len(self.background_tasks)
                if pending_count > 0:
                    logger.info(
                        f"[Gemini Image] 正在取消 {pending_count} 个后台生成任务..."
                    )
                    for task in self.background_tasks:
                        if not task.done():
                            task.cancel()
                    # 等待所有任务取消
                    await asyncio.gather(*self.background_tasks, return_exceptions=True)
                    logger.info("[Gemini Image] 所有后台任务已取消")

            # 清理图片缓存内存
            if hasattr(self, "recent_images"):
                total_images = sum(
                    len(images) for images in self.recent_images.values()
                )
                self.recent_images.clear()
                logger.info(
                    f"[Gemini Image] 已清理内存中的图片缓存 ({total_images} 张)"
                )

            # 清理生成器资源
            if hasattr(self, "generator") and self.generator:
                # 清理生成器的图片缓存
                if hasattr(self.generator, "image_cache"):
                    cache_count = len(self.generator.image_cache)
                    self.generator.image_cache.clear()
                    logger.info(f"[Gemini Image] 已清理生成器缓存 ({cache_count} 个)")

            logger.info("[Gemini Image] 插件已卸载")
        except Exception as e:
            logger.error(f"[Gemini Image] 清理资源时出错: {e}")
