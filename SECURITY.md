# 安全声明

## 运行用户建议

基于安全性角度考虑，不建议使用 root 等管理员类型账户执行任何命令，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统 umask 值为 0027 及以上，保障新增文件夹默认最高权限为 750，新增文件默认最高权限为 640。
- 建议用户对个人隐私数据、商业资产、源文件和算子开发过程中保存的各类文件等敏感内容做好权限控制等安全措施。本项目面向材料化学与流程工业场景，**分子模拟轨迹数据、材料晶体结构、反应路径参数、工艺控制模型权重等通常属于高敏感资产**，请对输入数据文件、中间计算结果与模型产物目录做好权限管控，设定的权限建议参考[附录 A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)。
- 算子运行时可能会缓存算子编译文件，存储在运行目录下的 `kernel_meta_*` 文件夹内，加快后续算子的调用速度，用户可根据需要自行对生成后的相关文件进行权限控制。
- 用户安装和使用过程需要做好权限控制，建议参考[附录 A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)进行设置。

## 构建安全声明

- 在源码编译安装本项目时，需要您自行编译，编译过程中会生成一些中间文件，建议您在编译完成后对中间文件做好权限控制，以保证文件安全。本项目的编译中间件与产物集中在以下位置，便于统一管控：
    - `<operator>/build/`：各算子 CMake 构建目录（cmake --build 产物）
    - `build_out/`：CI 跳过标记文件目录
- 本项目的 C++ 算子通过 CMake 统一编译。建议在算子 CMakeLists.txt 中启用安全编译选项：
    - `-fstack-protector-strong`：栈溢出保护
    - `-D_FORTIFY_SOURCE=2`：编译时缓冲区溢出检测
    - `-fPIC`：位置无关代码
    - `-Wl,-z,relro -Wl,-z,now`：GOT 只读与立即绑定
    - `-Wl,-z,noexecstack`：禁止可执行栈
- 安装 OPP `.run` 算子包会向 CANN 安装目录的 `opp/vendors/customize/` 写入文件，请确认执行账户对该目录具备必要且最小的写权限。

## 运行安全声明

- 建议用户结合运行环境资源状况编写对应算子调用脚本。若算子调用脚本与资源状况不匹配，如分子动力学模拟的数据帧数量超出内存容量限制、材料结构搜索的候选数超出显存容量等情况，可能会引发错误并导致进程意外退出。
- 算子在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看日志文件等方式。
- 本项目算子通过 PyTorch 方式调用时，可能会因为版本不匹配导致运行错误，具体请参考 [PyTorch 安全声明](https://gitcode.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。
- 本项目通过 `build.sh` 统一编译入口构建算子。构建脚本会读取 `ASCEND_TOOLKIT_HOME`、`SOC_VERSION` 等环境变量，这些环境变量会影响编译目标和产物位置，请勿在不可信环境中由外部输入直接设置。

## 公网地址声明

本项目**源码与配置文件**中包含的公网地址声明如下所示（Markdown 文档中用于跳转的社区链接不在此列）：

| 类型 | 开源代码地址 | 文件名 | 公网URL地址 | 用途说明 |
|---|---|---|---|---|
| 许可参考 | 不涉及 | CONTRIBUTING.md | [https://www.apache.org/licenses/](https://www.apache.org/licenses/) | Apache 2.0 许可证全文参考。本仓不依赖、不下载 |
| 社区帮助 | 不涉及 | CONTRIBUTING.md | [https://docs.gitcode.com/docs/help/home/org_project/pullrequests/](https://docs.gitcode.com/docs/help/home/org_project/pullrequests/) | GitCode PR 帮助文档链接，仅用于社区指引，不涉及依赖 |
| 社区帮助 | 不涉及 | CONTRIBUTING.md | [https://docs.gitcode.com/v1-docs/docs/pulls/cla/](https://docs.gitcode.com/v1-docs/docs/pulls/cla/) | GitCode CLA 帮助文档链接，仅用于社区指引 |
| 三方件声明 | 不涉及 | README.md / FAQ.md / CONTRIBUTING.md | [https://gitcode.com/ascend/pytorch](https://gitcode.com/ascend/pytorch) | torch_npu 上游获取地址，用于社区参考。PyTorch 由用户自行安装，本仓不下载、不分发 |

## 漏洞机制说明

[漏洞管理](https://gitcode.com/cann/community/blob/master/security/security.md)

## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型 | Linux权限参考最大值 |
|---|---|
| 用户主目录 | 750（rwxr-x---） |
| 程序文件（含脚本文件、库文件等） | 550（r-xr-x---） |
| 程序文件目录 | 550（r-xr-x---） |
| 配置文件 | 640（rw-r-----） |
| 配置文件目录 | 750（rwxr-x---） |
| 日志文件（记录完毕或者已经归档） | 440（r--r-----） |
| 日志文件（正在记录） | 640（rw-r-----） |
| 日志文件目录 | 750（rwxr-x---） |
| Debug文件 | 640（rw-r-----） |
| Debug文件目录 | 750（rwxr-x---） |
| 临时文件目录 | 750（rwxr-x---） |
| 维护升级文件目录 | 770（rwxrwx---） |
| 业务数据文件 | 640（rw-r-----） |
| 业务数据文件目录 | 750（rwxr-x---） |
| 密钥组件、私钥、证书、密文文件目录 | 700（rwx------） |
| 密钥组件、私钥、证书、加密密文 | 600（rw-------） |
| 加解密接口、加解密脚本 | 500（r-x------） |