---
name: no-secrets-in-memory
description: Memory 文件跟随项目 git commit，禁止写入 API key / password / token 等敏感信息
metadata:
  type: feedback
---

`.claude/memory/` 下所有文件会随项目 commit 入仓（公开仓库可被全网检索），写 memory 前**必须**确认不含：
- API key / access token / refresh token
- 数据库或服务密码
- 私有 endpoint URL 含鉴权参数
- SSH key / cert 私钥片段
- 任何 `.env` / secrets 文件原文

**Why:** 用户明确指出 memory 与代码同仓托管，泄露成本高且不可撤回（git 历史永久留存）。

**How to apply:** 涉及凭证场景时，用占位符（如 `<API_KEY>`、`$ENV_VAR`）或只描述"在 ~/.config/xxx 中"指明位置，不要把值本身写进 memory。
