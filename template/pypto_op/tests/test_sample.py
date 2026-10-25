"""PyPTO 模板测试：只验证包可导入，不代表真实算子。"""

import matchem_pypto_sample


def test_package_imports():
    assert matchem_pypto_sample.__version__ == "0.1.0"
