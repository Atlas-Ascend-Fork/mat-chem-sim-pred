# 文档中心

本目录是仓库唯一的文档入口。算子自身的算法/API/性能文档仍在各算子目录的 `docs/` 下，
本目录只放**跨算子**的通用文档。

## 目录

```text
docs/
└── zh/
    ├── README.md                     # 本页
    ├── QUICKSTART.md                 # 快速入门：环境、构建、测试、增量门禁
    ├── op_list.md                    # 全部算子清单与统计口径
    ├── appendix/
    │   └── dir_structure.md          # 目录结构：每个目录的职责与放置规则
    └── install/
        └── quick_install.md          # 环境安装与依赖
```

## 阅读路径

| 我想…… | 先看 |
|---|---|
| 在本机跑起来 | [QUICKSTART.md](QUICKSTART.md) → [install/quick_install.md](install/quick_install.md) |
| 知道仓里有哪些算子 | [op_list.md](op_list.md) |
| 新增一个算子 | [appendix/dir_structure.md](appendix/dir_structure.md) → [../../CONTRIBUTING.md](../../CONTRIBUTING.md) |
| 改 CI 或本地门禁 | [QUICKSTART.md](QUICKSTART.md) 的"增量门禁"一节 |

## 写作约定

- 中文文档放 `docs/zh/`；当前不维护英文副本（CANN 算子仓惯例，opbase 的 `docs/en` 是框架库特例）。
- 一页只讲一件事；命令必须能直接复制执行。
- 涉及数量的表述以 [op_list.md](op_list.md) 为准，不在多处重复维护。
- 新增文档时同步更新本页目录，避免"文档存在但没人找得到"。
