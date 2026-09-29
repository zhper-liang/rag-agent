使用 DISKANN 索引前，需要在 milvus.yaml 里把 queryNode.enableDisk 设为 true，否则加载集合会报错[1]。

修改配置前先备份原文件，改完后重启所有组件才会生效[1]。部署环境要求 Docker 20.10 以上、内存不少于 16 GB[2]。

升级时还要确认新版本与现有索引类型兼容，DISKANN 从 2.3 起支持 mmap[3]。
