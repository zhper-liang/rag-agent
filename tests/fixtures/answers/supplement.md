磁盘索引的数据放在 localStorage.path 指定的目录，建议挂载 NVMe 盘[1]。

**补充说明（以下内容不来自资料）**

NVMe 盘的随机读延迟通常比 SATA SSD 低一个数量级，DISKANN 查询时大量随机读，所以对磁盘延迟比较敏感。
