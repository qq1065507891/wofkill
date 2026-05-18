# 生产部署指南

本文档描述 Werewolf Agent 的生产部署流程，包括 Docker Compose 编排、环境变量、存储后端、模型路由、监控和备份策略。

---

## 1. 快速启动（Docker Compose）

### 1.1 最小启动

启动 API 服务 + PostgreSQL/pgvector：

```bash
# 复制并填写环境变量
cp .env.example .env
# 编辑 .env，填写 API Key

docker compose up -d
```

- API 监听端口默认 `18000`，通过 `WEREWOLF_API_PORT` 环境变量修改。
- 观战台 Dashboard：http://localhost:18000
- PostgreSQL 使用 `pgvector/pgvector:pg16` 镜像，同时提供关系存储和向量存储。

### 1.2 含 Redis 的完整启动

```bash
docker compose --profile with-redis up -d
```

Redis 用于多进程分布式锁和运行时状态追踪（`RedisRuntimeExecutor`）。单进程部署不需要 Redis。

### 1.3 自定义端口

```bash
WEREWOLF_API_PORT=8000 docker compose up -d
```

### 1.4 生产 Profile

```bash
docker compose --profile production-adapters up -d
```

`production-adapters` profile 同时启用 Redis。

---

## 2. 环境变量

所有配置通过 `.env` 文件或环境变量注入，**不要在代码或配置文件中硬编码密钥**。

### 2.1 LLM Provider 密钥

| 变量 | 说明 |
|------|------|
| `ANTHROPIC_API_KEY` | Anthropic 兼容 API 密钥（当前用于 MiniMax-M2.7） |
| `ANTHROPIC_BASE_URL` | Anthropic 兼容 API 基地址，默认 `https://api.minimaxi.com/anthropic` |
| `GLM_API_KEY` | GLM/OpenAI 兼容 API 密钥（可选） |
| `GLM_BASE_URL` | GLM API 基地址，默认 `https://open.bigmodel.cn/api/paas/v4` |
| `OPENAI_API_KEY` | OpenAI API 密钥（可选） |
| `OPENAI_BASE_URL` | OpenAI API 基地址 |

### 2.2 存储配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEREWOLF_STORAGE_BACKEND` | 空（内存） | `memory` / `sqlite` / `postgres` |
| `WEREWOLF_DB_PATH` | `data/wofkill.db` | SQLite 数据库文件路径 |
| `POSTGRES_DSN` | 空 | PostgreSQL 连接串，格式：`postgresql://user:pass@host:5432/dbname` |

### 2.3 向量存储配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEREWOLF_VECTOR_BACKEND` | 空（auto） | `auto` / `local` / `embedding` / `siliconflow` / `pgvector` |
| `PGVECTOR_DSN` | 空 | pgvector 连接串，通常与 `POSTGRES_DSN` 相同 |

### 2.4 RAG 配置

| 变量 | 说明 |
|------|------|
| `SILICONFLOW_API_KEY` | SiliconFlow 嵌入和 Reranker API 密钥 |
| `SILICONFLOW_BASE_URL` | SiliconFlow API 基地址，默认 `https://api.siliconflow.cn` |

### 2.5 功能开关与运行时

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEREWOLF_USE_LLM_AGENTS` | `1` | 设为 `0` 禁用真实 LLM 调用（使用 stub agent） |
| `WEREWOLF_MODEL_CONFIG` | `config/models.yaml` | 模型路由配置文件路径 |
| `WEREWOLF_AUTH_SECRET` | `wofkill-dev-key` | HMAC session 签名密钥，生产环境务必替换 |
| `REDIS_URL` | 空 | Redis 连接串，如 `redis://localhost:6379/0` |
| `WEREWOLF_API_PORT` | `18000` | Docker Compose 宿主机映射端口 |

### 2.6 Docker Compose 内置环境变量

`docker-compose.yml` 中为 API 容器预设了以下环境变量，可在 `.env` 中覆盖：

```yaml
environment:
  - WEREWOLF_STORAGE_BACKEND=postgres
  - POSTGRES_DSN=postgresql://wofkill:wofkill-dev@postgres:5432/wofkill
  - WEREWOLF_VECTOR_BACKEND=pgvector
  - PGVECTOR_DSN=postgresql://wofkill:wofkill-dev@postgres:5432/wofkill
  - WEREWOLF_USE_LLM_AGENTS=${WEREWOLF_USE_LLM_AGENTS:-1}
  - WEREWOLF_MODEL_CONFIG=config/models.yaml
  - WEREWOLF_AUTH_SECRET=${WEREWOLF_AUTH_SECRET:-wofkill-dev-key}
```

---

## 3. 存储后端

### 3.1 内存存储（默认，仅开发）

不设置 `WEREWOLF_STORAGE_BACKEND` 即使用内存存储。进程退出后数据全部丢失。适用于快速验证和单元测试。

### 3.2 SQLite（本地持久化开发）

```bash
WEREWOLF_STORAGE_BACKEND=sqlite
WEREWOLF_DB_PATH=data/wofkill.db
```

- 使用 Python 标准库 `sqlite3`，零外部依赖。
- 数据文件存储在 `data/wofkill.db`，通过 Docker volume `werewolf-data` 持久化。
- 适用于单机开发和调试。

### 3.3 PostgreSQL（生产推荐）

```bash
WEREWOLF_STORAGE_BACKEND=postgres
POSTGRES_DSN=postgresql://wofkill:your-password@postgres:5432/wofkill
```

- 使用 JSONB 存储游戏状态，保持与 SQLite/内存相同的 Repository 接口。
- 支持事件追加、死亡记录、模型用量、评测结果和配置快照的持久化。
- Docker Compose 默认使用 `pgvector/pgvector:pg16` 镜像，同时提供 pgvector 扩展。
- **生产环境务必修改默认密码**。

---

## 4. LLM Provider 配置

### 4.1 配置文件结构

模型路由通过 `config/models.yaml` 配置，分为三层：

```yaml
# 第一层：模型参数预设
model_profiles:
  minimax_m27_default:
    provider: anthropic
    model: MiniMax-M2.7
    temperature: 0.5
    max_tokens: 1024
    top_p: 0.9
    timeout: 60

# 第二层：任务路由配置
llm_profiles:
  minimax_default:
    default:
      provider: anthropic
      model_profile: minimax_m27_default
    tasks:
      reflection:
        provider: anthropic
        model_profile: minimax_m27_reflection
      speech:
        provider: anthropic
        model_profile: minimax_m27_default
      deception:
        provider: anthropic
        model_profile: minimax_m27_fast
      night_action:
        provider: anthropic
        model_profile: minimax_m27_default
    fallback:
      provider: anthropic
      model_profile: minimax_m27_fast

# 第三层：玩家-人格-模型绑定
players:
  p01:
    persona_id: logic_leader
    llm_profile: minimax_default
  # ... p02-p12
  judge:
    persona_id: judge
    llm_profile: minimax_default
```

### 4.2 模型路由说明

- **model_profiles**：定义模型参数模板（provider、模型名、温度、token 上限、超时）。
- **llm_profiles**：将不同任务（reflection/speech/deception/night_action）路由到不同模型配置，并提供 fallback。
- **players**：为每个玩家位（p01-p12）和 judge 绑定人格和 LLM 配置文件。
- API Key 仅通过环境变量提供，不写入配置文件。

### 4.3 代码中使用

```python
from werewolf_agent.model_gateway.router import ModelRouter

router = ModelRouter.from_yaml(
    "config/models.yaml",
    register_env_providers=True,
)
```

`register_env_providers=True` 会从环境变量自动注册已配置的 provider。

### 4.4 内置 Provider

| Provider | 兼容协议 | 用途 |
|----------|----------|------|
| `anthropic` | Anthropic Messages API | 当前主用，指向 MiniMax |
| `glm` | OpenAI Chat Completions | 保留扩展 |
| `openai` | OpenAI Chat Completions | 保留扩展 |

---

## 5. 监控与健康检查

### 5.1 API 健康检查

Docker Compose 配置了自动健康检查：

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/games')"]
  interval: 10s
  timeout: 5s
  retries: 3
```

检查 API 容器内 `GET /games` 端点是否正常响应。

### 5.2 手动检查

```bash
# 容器内（通过 Docker healthcheck）
curl http://localhost:8000/games

# 宿主机（通过映射端口）
curl http://localhost:18000/games
```

返回 `200 OK` 且为 JSON 数组表示服务正常。

### 5.3 日志配置

```bash
# 查看 API 日志
docker compose logs api -f

# 查看 PostgreSQL 日志
docker compose logs postgres -f

# 查看 Redis 日志
docker compose logs redis -f

# 查看所有服务日志
docker compose logs -f
```

Python 应用日志级别可通过标准 logging 配置调整。

---

## 6. 备份策略

### 6.1 PostgreSQL 备份

```bash
# 全量备份
docker compose exec postgres pg_dump -U wofkill wofkill > backup_$(date +%Y%m%d_%H%M%S).sql

# 仅备份游戏数据
docker compose exec postgres pg_dump -U wofkill -t games -t events -t deaths wofkill > games_backup.sql

# 恢复
cat backup.sql | docker compose exec -T postgres psql -U wofkill wofkill
```

### 6.2 SQLite 备份

```bash
# 直接拷贝文件（需先停止写入）
cp data/wofkill.db backup_$(date +%Y%m%d_%H%M%S).db

# 或使用 SQLite 安全导出
sqlite3 data/wofkill.db ".backup backup.db"
```

### 6.3 游戏状态序列化

游戏状态通过 `_serialize_game_state()` 序列化为 JSON，可通过 API 导出：

```bash
# 获取完整 replay 数据
curl http://localhost:18000/games/{game_id}/replay > replay_{game_id}.json

# 获取事件时间线
curl http://localhost:18000/games/{game_id}/timeline > timeline_{game_id}.json
```

### 6.4 建议的备份频率

| 场景 | 频率 | 方式 |
|------|------|------|
| 开发环境 | 按需 | SQLite 文件拷贝 |
| 生产环境 | 每日 | pg_dump + 定时任务 |
| 重要对局 | 对局结束后 | API replay 导出 |

---

## 7. 扩展与性能

### 7.1 Redis 分布式锁

多进程/多实例部署时，启用 `RedisRuntimeExecutor` 替代默认的 `LocalRuntimeExecutor`：

- 每个 game_id 对应一个分布式锁，TTL 默认 5 分钟。
- 提供 `acquire_lock` / `release_lock` / `refresh_lock` 接口。
- Redis 不可用时优雅降级（返回 False/None），不会阻塞主流程。

```bash
# 启动含 Redis 的环境
docker compose --profile with-redis up -d

# 设置环境变量
REDIS_URL=redis://redis:6379/0
```

### 7.2 PostgreSQL 连接池

当前 `PostgresGameRepository` 使用单连接模式。生产环境如需连接池：

- 可在 `PostgresGameRepository.__init__` 中替换为 `psycopg_pool.ConnectionPool`。
- Docker Compose 的 PostgreSQL 默认 `max_connections=100`，可通过 `POSTGRES_MAX_CONNECTIONS` 环境变量调整。

### 7.3 静态文件服务

Dashboard 静态文件通过 FastAPI `StaticFiles` 中间件直接服务，适合开发和小规模部署。生产环境建议：

- 在 API 前放置 Nginx/Caddy 反向代理。
- 由 Nginx 直接服务 `/static/` 路径下的文件。
- API 仅处理动态请求。

示例 Nginx 配置：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /static/ {
        root /app/werewolf_agent/ui;
        expires 7d;
    }

    location / {
        proxy_pass http://127.0.0.1:18000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 7.4 资源估算

| 组件 | 最小 | 推荐 |
|------|------|------|
| API | 1 CPU / 512MB | 2 CPU / 1GB |
| PostgreSQL | 1 CPU / 256MB | 1 CPU / 512MB |
| Redis | 0.5 CPU / 128MB | 0.5 CPU / 256MB |

---

## 8. 安全注意事项

1. **替换默认密钥**：生产环境必须修改 `WEREWOLF_AUTH_SECRET` 和 PostgreSQL 密码。
2. **不要暴露 PostgreSQL 端口**：生产环境移除 `docker-compose.yml` 中 PostgreSQL 的 `ports` 映射，仅允许容器内网访问。
3. **API Key 管理**：所有 LLM provider 密钥通过 `.env` 文件管理，不要提交到版本控制。
4. **网络隔离**：建议将 API、数据库、Redis 放在同一 Docker 网络，仅暴露 API 端口。
5. **HTTPS**：生产环境通过反向代理（Nginx/Caddy）提供 TLS 终止。
