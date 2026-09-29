改完配置后用 docker compose 重启服务，再确认各容器状态正常[1]：

```bash
docker compose up -d
docker compose ps
```

如果容器反复重启，先检查内存是否满足 16 GB 的要求[2]。
