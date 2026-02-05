# qq-group-files-sync-python

在多个QQ群和本地储存之间同步群文件的小工具，可选生成一个可浏览的静态页面。

### 本工具的诞生离不开这些优秀的前辈

- [qq-group-files-sync](https://github.com/sealdice/qq-group-files-sync)
- [LuckyLilliaBot](https://github.com/LLOneBot/LuckyLilliaBot)

### 另外感谢这些给予我知识和鼓励的老师们

- [Gemini](https://gemini.google.com)
- [ChatGPT](https://chatgpt.com)

功能：

- 群文件下载
- 增量同步
- 生成展示界面
- 通过 OneBot11 **正向 WS 连接** 与协议端通信

## 使用

1) 准备一个 OneBot11 协议端（开发&测试使用 LuckyLilliaBot）。
2) 在协议端开启 **OneBot11 WebSocket 服务器**，并确认地址（例如：`ws://127.0.0.1:3001`）。
3) 使用 [uv](https://docs.astral.sh/uv/getting-started/installation) 安装依赖：
4) 在项目目录根据[配置示例](#配置示例)示例创建 `config.toml`（当然，首次运行会自动生成一份带注释的示例配置）
5) `uv run main.py [OPTIONS] COMMAND [ARGS]...`
6) Enjoy!

```bash
# 安装依赖
uv sync

# 查看帮助
uv run main.py --help

# 拉取所有（增量备份）
uv run main.py pull all

# 只拉取某一个群
uv run main.py pull QQ-Group:123456

# 如果希望同步完成后生成/更新展示页面，加 --web（或 -w）
uv run main.py pull all --web

# 推送（增量：只补齐远端缺失的文件）
uv run main.py push all

# 推送单个群
uv run main.py push QQ-Group:123456

# 进入交互等待模式（在群里发指令控制同步）
uv run main.py watch
```

日志默认写入：

- `./logs/main.log`：全部日志（含 debug）
- `./logs/error.log`：警告与错误（warn/error）

控制台输出级别由 `logLevel` 控制；httpx/httpcore 的请求明细会被视为 debug 级别。

## 交互模式指令

在群内发送：

- `.同步当前`

  - 同步当前群的群文件，完成后自动生成展示页面
- `.同步文件 QQ-Group:群号`

  - 指定一个群进行同步（机器人账号应在目标群内）
- `.同步全部`

  - 同步 `config.toml` 中设置的所有群
- `.展示页面`

  - 强制重新生成展示页面

## 同步过程说明

同步启动后，会进行增量预测，只进行增量更新。控制台会输出类似：

```text
========== 同步预测结果 ==========
群内文件: 7 个文件夹 / 359 个文件 / 总大小 1.8 GB
当前已存: 7 个文件夹 / 200 个文件 / 大小 1011.1 MB
需要更新: 159 个文件 / 下载 799.2 MB
需要创建: 0 个文件夹
需要删除: 0 个文件 / 0 个文件夹 / 释放 0 B
================================
```

展示页面默认输出到数据目录下的 `list.html`。
（命令行 pull 默认不生成页面；如需生成请加 `--web`；交互模式下会自动生成。）

push 命令不会修改远端已有文件：

- 仅对比“远端文件列表 vs 本地文件列表”
- 只上传远端缺失的文件

## 配置示例

```toml
# OneBot11 正向 WS 地址
[onebot11]
ws_url = "ws://127.0.0.1:3001"
access_token = ""

[file_system]
local_path = "./data"

log_file = "./logs/main.log"
log_level = "info"

[[groups]]
id = "QQ-Group:123456"
alias = "名称"
description = "介绍"

[web]
title = "群文件导航"
base_url = ""
dashboard_file = "list.html"
```

## 其他

- 小文件较多时可考虑并发下载（`--concurrency`参数）。
