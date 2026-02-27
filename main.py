"""
Gemini Image Generation Plugin
功能：文生图、图生图、白名单管理、网络连通性测试、模型切换、预设管理
安全更新：增加 API Key 自动脱敏，防止报错信息泄露密钥
优化：移除全局关键词拦截，防止误触
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Coroutine
from typing import Any

import aiohttp # 用于测试网络连接
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.utils.io import download_image_by_url, save_temp_img

from .gemini_generator import GeminiImageGenerator


@pydantic_dataclass
class GeminiImageGenerationTool(FunctionTool[AstrAgentContext]):
    """统一的图像生成工具"""

    name: str = "gemini_generate_image"
    description: str = "使用 Gemini 模型生成或修改图片。仅在用户明确要求生成图像时使用此工具。"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图提示词",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比",
                    "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                },
                "resolution": {
                    "type": "string",
                    "description": "分辨率(仅 Pro 模型支持)",
                    "enum": ["1K", "2K", "4K"],
                },
                "avatar_references": {
                    "type": "array",
                    "description": "参考头像(self/sender/qq号)",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    plugin: object | None = None

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        if not (prompt := kwargs.get("prompt", "")):
            return "请提供提示词"

        plugin = self.plugin
        if not plugin:
            return "❌ 插件未初始化"

        event = None
        if hasattr(context, "context") and isinstance(context.context, AstrAgentContext):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            return "❌ 无法获取上下文"

        # --- 白名单拦截 ---
        user_id = event.unified_msg_origin
        group_id = getattr(event.message_obj, "group_id", None)
        sender_id = str(event.message_obj.sender.user_id) if event.message_obj.sender else user_id
        
        if not plugin._check_permission(sender_id, str(group_id) if group_id else None):
            logger.info(f"[Gemini Image] LLM工具调用拦截: {sender_id}")
            return "❌ 拒绝：您没有权限使用此功能(白名单拦截)"
        # ----------------

        if not plugin.generator.api_keys:
            return "❌ 未配置 API Key"

        # 获取参考图
        images_data = await plugin._get_reference_images_for_tool(event)
        
        # 处理头像引用
        for ref in kwargs.get("avatar_references", []):
            if not isinstance(ref, str): continue
            uid = None
            if ref == "self": uid = str(event.get_self_id())
            elif ref == "sender": uid = str(event.get_sender_id() or event.unified_msg_origin)
            else: uid = ref
            
            if uid and (avatar := await plugin.get_avatar(uid)):
                images_data.append((avatar, "image/jpeg"))

        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]
        
        # 修复参数名匹配问题
        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                target=event.unified_msg_origin, # 对应 target
                refs=images_data or None,        # 对应 refs
                ar=kwargs.get("aspect_ratio", plugin.default_aspect_ratio), # 对应 ar
                res=kwargs.get("resolution", plugin.default_resolution),    # 对应 res
                tid=task_id,                     # 对应 tid
            )
        )
        return "✅ 生图任务已启动，请稍候..."


class GeminiImagePlugin(Star):
    """Gemini 图像生成插件"""

    AVAILABLE_MODELS = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-3-pro-image-preview",
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or AstrBotConfig()
        
        # 初始化基础属性
        self.api_keys = []
        self.base_url = ""
        self.proxy = None
        self.user_request_timestamps = {}
        self.background_tasks = set()

        # 加载配置
        self._load_config()

        # 初始化生成器
        self.generator = GeminiImageGenerator(
            api_keys=self.api_keys,
            base_url=self.base_url,
            model=self.model,
            api_type=self.api_type,
            timeout=self.timeout,
            max_retry_attempts=self.max_retry_attempts,
            proxy=self.proxy,
            safety_settings=self.safety_settings,
        )

        self._generation_semaphore = asyncio.Semaphore(self.max_concurrent_generations)

        if self.enable_llm_tool:
            self.context.add_llm_tools(GeminiImageGenerationTool(plugin=self))

        logger.info(f"[Gemini Image] 插件加载完成 | 模型: {self.model} | 代理: {self.proxy or '无'}")

    def _load_config(self):
        """加载配置"""
        api_config = self.config.get("api_config", {})
        gen_config = self.config.get("generate_config", {})
        wl_config = self.config.get("whitelist_config", {})
        perm_conf = self.config.get("permission_config", {})

        # 白名单
        self.enable_whitelist = wl_config.get("enable_whitelist", False)
        self.allowed_groups = [str(x) for x in wl_config.get("allowed_groups", [])]
        self.allowed_users = [str(x) for x in wl_config.get("allowed_users", [])]

        # 拦截配置 (从 main.bak.py 迁移)
        self.perm_no_permission_reply = perm_conf.get("no_permission_reply", "❌ 您没有权限使用此功能")
        self.perm_silent = perm_conf.get("silent_on_no_permission", False)
        self.perm_intercept_keywords = perm_conf.get("intercept_keywords", ["画", "绘", "图", "draw", "image", "photo", "generate", "生图"])

        # 基础 API 配置
        self.api_type = api_config.get("api_type", "gemini")
        provider_id = api_config.get("provider_id", "")
        
        # 1. 优先读取插件配置中的代理
        self.proxy = api_config.get("proxy", "").strip() or None

        # 2. 如果使用系统提供商
        use_system = api_config.get("use_system_provider", True)
        loaded_system = False
        if use_system and provider_id:
            loaded_system = self._load_provider_config(provider_id)
        
        # 3. 如果没用系统提供商，或者系统提供商加载失败，加载手动配置
        if not loaded_system:
            if use_system: logger.warning("[Gemini Image] 系统提供商加载失败，尝试使用手动配置")
            self._load_manual_config(api_config)

        self.model = self._load_model_config()

        # 生图参数
        self.timeout = gen_config.get("timeout", 300)
        self.default_aspect_ratio = gen_config.get("default_aspect_ratio", "1:1")
        self.default_resolution = gen_config.get("default_resolution", "1K")
        self.max_retry_attempts = gen_config.get("max_retry_attempts", 3)
        self.safety_settings = gen_config.get("safety_settings", "BLOCK_NONE")
        self.max_image_size_mb = gen_config.get("max_image_size_mb", 10)
        self.max_requests_per_minute = gen_config.get("max_requests_per_minute", 3)
        self.debug_mode = gen_config.get("debug_mode", False)
        
        mc = gen_config.get("max_concurrent_generations", 3)
        self.max_concurrent_generations = max(1, min(mc, 10))

        self.enable_llm_tool = self.config.get("enable_llm_tool", True)
        self.presets = self._load_presets()

    def _load_provider_config(self, provider_id: str) -> bool:
        """从系统提供商加载"""
        provider = self.context.get_provider_by_id(provider_id)
        if not provider: return False
        
        cfg = getattr(provider, "provider_config", {}) or {}
        
        # 提取 Keys
        keys = cfg.get("api_key") or cfg.get("key") or cfg.get("keys") or cfg.get("access_token")
        if not keys: return False
        self.api_keys = [keys] if isinstance(keys, str) else keys

        # 提取 Base URL
        base = getattr(provider, "api_base", None) or cfg.get("api_base") or cfg.get("api_base_url")
        self.base_url = self._clean_base_url(base or "https://generativelanguage.googleapis.com")

        if not self.proxy:
            sys_proxy = getattr(provider, "proxy", None) or cfg.get("proxy")
            if sys_proxy:
                self.proxy = sys_proxy
                logger.info(f"[Gemini Image] 继承系统代理: {self.proxy}")

        return True

    def _load_manual_config(self, api_config):
        keys = api_config.get("api_key", [])
        self.api_keys = [k for k in keys if k] if isinstance(keys, list) else [keys] if keys else []
        self.base_url = self._clean_base_url(api_config.get("base_url", "https://generativelanguage.googleapis.com"))

    def _load_model_config(self) -> str:
        model = self.config.get("api_config", {}).get("model", "gemini-2.5-flash-image")
        if model == "自定义模型":
            return self.config.get("api_config", {}).get("custom_model", "")
        return model

    def _clean_base_url(self, url: str) -> str:
        if not url: return ""
        url = url.rstrip("/")
        if "/v1" in url: url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_presets(self) -> dict:
        raw = self.config.get("presets", [])
        presets = {}
        for p in raw:
            if isinstance(p, str) and ":" in p:
                k, v = p.split(":", 1)
                presets[k.strip()] = v.strip()
        return presets

    def _check_permission(self, user_id: str, group_id: str | None = None) -> bool:
        if not self.enable_whitelist: return True
        if str(user_id) in self.allowed_users: return True
        if group_id and str(group_id) in self.allowed_groups: return True
        return False

    def _check_rate_limit(self, user_id: str) -> bool:
        now = time.time()
        ts = self.user_request_timestamps.setdefault(user_id, [])
        ts = [t for t in ts if now - t < 60]
        self.user_request_timestamps[user_id] = ts
        if len(ts) >= self.max_requests_per_minute: return False
        ts.append(now)
        return True

    # --- 安全脱敏函数 ---
    def _sanitize_error_msg(self, error_msg: str) -> str:
        """检查并隐藏敏感信息 (API Key)"""
        if not error_msg: return "未知错误"
        msg_str = str(error_msg)
        # 只要包含敏感特征，立即返回通用错误，不输出原始信息
        if "api_key" in msg_str or "AIza" in msg_str:
            # 记录完整日志到后台，方便管理员排查
            logger.error(f"[Gemini Image] 安全拦截敏感报错: {msg_str}")
            return "⚠️ API 鉴权失败 (Key 无效或已暂停)。\n(为保护安全，详细报错已隐藏，请查看后台日志)"
        return msg_str

    # 已移除 intercept_drawing_request 方法，防止误触

    @filter.command("gemini_test")
    async def test_connectivity(self, event: AstrMessageEvent):
        """测试 Gemini API 连通性"""
        user_id = event.unified_msg_origin
        sender_id = str(event.message_obj.sender.user_id) if event.message_obj.sender else user_id
        if self.enable_whitelist and not self._check_permission(sender_id):
            yield event.plain_result("❌ 无权执行测试")
            return

        proxy_status = f"当前配置代理: {self.proxy}" if self.proxy else "当前未配置代理 (直连)"
        yield event.plain_result(f"🔄 开始测试网络连通性...\n{proxy_status}")

        target_url = "https://generativelanguage.googleapis.com"
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, proxy=self.proxy, timeout=10) as resp:
                    latency = (time.time() - start_time) * 1000
                    status = resp.status
                    
                    if status == 200 or status == 404: 
                        msg = (f"✅ **连接成功！**\n"
                               f"目标: Google API\n"
                               f"状态码: {status}\n"
                               f"延迟: {latency:.2f}ms\n"
                               f"代理生效: {'是' if self.proxy else '否'}")
                    else:
                        msg = (f"⚠️ **连接异常**\n"
                               f"状态码: {status}\n"
                               f"提示: 能连上但返回了错误。")
                        
        except asyncio.TimeoutError:
            msg = (f"❌ **连接超时 (Timeout)**\n"
                   f"原因: 10秒内无法连接到 Google 服务器。\n"
                   f"建议: 请检查代理地址是否正确，或代理软件是否允许外部连接。")
        except Exception as e:
            msg = f"❌ **连接失败**\n错误: {str(e)}"

        yield event.plain_result(msg)

    @filter.command("生图调试")
    async def debug_switch(self, event: AstrMessageEvent):
        """开关调试模式"""
        self.debug_mode = not self.debug_mode
        self.config["generate_config"]["debug_mode"] = self.debug_mode
        self.config.save_config()
        yield event.plain_result(f"🔧 调试模式: {'✅ 开启' if self.debug_mode else '⛔ 关闭'}")

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """生图模型管理指令"""
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = event.message_obj.group_id or ""
        
        if not self._check_permission(user_id, group_id):
            yield event.plain_result("❌ 您没有权限使用此功能")
            return

        if not model_index:
            model_list = ["📋 可用模型列表:"]
            for idx, model in enumerate(self.AVAILABLE_MODELS, 1):
                marker = " ✓" if model == self.model else ""
                model_list.append(f"{idx}. {model}{marker}")

            model_list.append(f"\n当前使用: {self.model}")
            yield event.plain_result("\n".join(model_list))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(self.AVAILABLE_MODELS):
                new_model = self.AVAILABLE_MODELS[index]
                self.model = new_model
                self.generator.model = new_model
                # 保存到分组配置
                if "api_config" not in self.config:
                    self.config["api_config"] = {}
                self.config["api_config"]["model"] = new_model
                self.config.save_config()
                yield event.plain_result(f"✅ 模型已切换: {new_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """预设管理指令"""
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = event.message_obj.group_id or ""
        
        if not self._check_permission(user_id, group_id):
            yield event.plain_result("❌ 您没有权限使用此功能")
            return

        # user_id 已经是正确的ID了
        masked_uid = (
            user_id[:4] + "****" + user_id[-4:] if len(user_id) > 8 else user_id
        )

        message_str = (event.message_str or "").strip()
        logger.info(
            f"[Gemini Image] 收到预设指令 - 用户: {masked_uid}, 内容: {message_str}"
        )

        parts = message_str.split(maxsplit=1)

        cmd_text = ""
        if len(parts) > 1:
            cmd_text = parts[1].strip()

        if not cmd_text:
            if not self.presets:
                yield event.plain_result("📋 当前没有预设")
                return

            preset_list = ["📋 预设列表:"]
            for idx, (name, prompt) in enumerate(self.presets.items(), 1):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            parts = cmd_text[3:].split(":", 1)
            if len(parts) == 2:
                name, prompt = parts
                self.presets[name.strip()] = prompt.strip()
                # 保存
                self.config["presets"] = [f"{k}:{v}" for k, v in self.presets.items()]
                self.config.save_config()
                yield event.plain_result(f"✅ 预设已添加: {name.strip()}")
            else:
                yield event.plain_result("❌ 格式错误: /预设 添加 名称:内容")

        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if name in self.presets:
                del self.presets[name]
                self.config["presets"] = [f"{k}:{v}" for k, v in self.presets.items()]
                self.config.save_config()
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 未找到预设: {name}")
        else:
             yield event.plain_result("❌ 格式错误，请使用：\n/预设\n/预设 添加 名称:内容\n/预设 删除 名称")

    @filter.command("生图")
    async def generate_image(self, event: AstrMessageEvent):
        user_id = event.unified_msg_origin
        group_id = getattr(event.message_obj, "group_id", None)
        sender_id = str(event.message_obj.sender.user_id) if event.message_obj.sender else user_id

        #这里修改无权限回复文本
        if not self._check_permission(sender_id, str(group_id) if group_id else None):
            yield event.plain_result("❌ 无权使用喵！")
            return

        if not self._check_rate_limit(user_id):
            yield event.plain_result("❌ 太快了喵！等一等喵！")
            return

        msg_str = (event.message_str or "").strip()
        parts = msg_str.split(maxsplit=1)
        if len(parts) < 2: return 
        
        raw_prompt = parts[1].strip()
        prompt = raw_prompt
        ar = self.default_aspect_ratio
        res = self.default_resolution
        
        # 预设匹配
        preset_key = raw_prompt.split()[0]
        for k, v in self.presets.items():
            if k.lower() == preset_key.lower():
                try:
                    if v.strip().startswith("{"):
                        d = json.loads(v)
                        prompt = d.get("prompt", prompt)
                        ar = d.get("aspect_ratio", ar)
                        res = d.get("resolution", res)
                    else:
                        prompt = v
                    extra = raw_prompt[len(preset_key):].strip()
                    if extra: prompt += " " + extra
                except:
                    prompt = v
                break

        if not prompt:
            yield event.plain_result("❌ 提示词呢喵！")
            return

        yield event.plain_result("🎨 正在生成...")

        refs = await self._fetch_images(event)
        task_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:6]
        logger.info(f"[Gemini Image] 任务[{task_id}] | Proxy: {self.proxy}")
        
        self.create_background_task(
            self._generate_and_send_image_async(
                prompt, event.unified_msg_origin, refs, ar, res, task_id
            )
        )

    async def _get_reference_images_for_tool(self, event: AstrMessageEvent) -> list[tuple[bytes, str]]:
        """为 LLM 工具获取参考图片 (引用回复或当前消息)"""
        return await self._fetch_images(event)

    async def _fetch_images(self, event: AstrMessageEvent):
        imgs = []
        if not event.message_obj.message: return imgs
        for comp in event.message_obj.message:
            url = None
            if isinstance(comp, Comp.Image): url = comp.url or comp.file
            elif isinstance(comp, Comp.Reply) and comp.chain:
                for c in comp.chain:
                    if isinstance(c, Comp.Image): url = c.url or c.file
            if url:
                d = await self._download_img(url)
                if d: imgs.append(d)
            if isinstance(comp, Comp.At) and comp.qq != "all":
                if str(comp.qq) != str(event.get_self_id()):
                   d = await self.get_avatar(str(comp.qq))
                   if d: imgs.append((d, "image/jpeg"))
        return imgs

    async def _download_img(self, url):
        try:
            # 优先检查本地文件 (兼容旧版行为)
            if os.path.exists(url) and os.path.isfile(url):
                with open(url, "rb") as f:
                    data = f.read()
            else:
                # 否则尝试下载
                path = await download_image_by_url(url)
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        data = f.read()
                else:
                    return None

            if data and len(data) <= self.max_image_size_mb * 1024 * 1024:
                return (data, "image/jpeg") 
        except: pass
        return None

    @staticmethod
    async def get_avatar(uid):
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={uid}&spec=640"
        path = await download_image_by_url(url)
        if path:
            with open(path, "rb") as f: return f.read()
        return None

    def create_background_task(self, coro):
        t = asyncio.create_task(coro)
        self.background_tasks.add(t)
        t.add_done_callback(self.background_tasks.discard)

    async def _generate_and_send_image_async(self, prompt, target, refs, ar, res, tid):
        if ar == "自动": ar = None
        async with self._generation_semaphore:
            try:
                imgs, err = await self.generator.generate_image(
                    prompt=prompt, images_data=refs, aspect_ratio=ar, image_size=res, task_id=tid
                )
                
                if err:
                    safe_err = self._sanitize_error_msg(str(err))
                    msg = f"❌ 失败: {safe_err}"
                    await self.context.send_message(target, MessageChain().message(msg))
                    return

                chain = MessageChain()
                for i in imgs:
                    p = save_temp_img(i)
                    chain.file_image(p)
                await self.context.send_message(target, chain)

            except Exception as e:
                logger.error(f"Generate Error: {e}", exc_info=True)
                safe_e = self._sanitize_error_msg(str(e))
                msg = f"❌ 发生内部错误: {safe_e}"
                await self.context.send_message(target, MessageChain().message(msg))

    async def terminate(self):
        if self.generator: await self.generator.close_session()
