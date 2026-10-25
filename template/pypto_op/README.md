# PyPTO 算子模板

这是一个不会被 `build.sh` 发现的模板，用来展示 PyPTO 后端的目录契约。

```text
<op>/
├── op.yaml          # backend: pypto + build/test/package 命令
├── pyproject.toml   # Python 包元数据
├── matchem_pypto_sample/
│   └── __init__.py
└── tests/
    └── test_sample.py
```

要点：

- `op.yaml` 的 `backend: pypto` 让 `build.sh` 走 PyPTO 分派，而不是 CMake；
- `build` / `test` / `package` 是 shell 命令，在算子目录内执行；
- PyPTO 算子不需要 `msopgen/*.json`；R7 只对 `backend: ascendc` 生效；
- 如果 PyPTO 最终产出的是 custom opp，而不是 Python wheel，把 `package` 命令换成
  对应的打包命令，并在 `build.sh --package` 里复用 Ascend C 的包消费方式。
