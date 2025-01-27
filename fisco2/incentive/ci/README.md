### CI/CD

CI/CD = 持续集成(Continuous Integration) + 持续部署(Continuous Deployment)

举个例子：假设你在开发一个网上商城
1. **持续集成(CI)**：
   ```
   小明改了登录功能
   小红改了购物车功能
   小张改了支付功能
   ```
   - 每个人提交代码时，CI系统会自动：
     - 运行所有测试
     - 检查代码质量
     - 进行安全扫描
   - 如果测试失败，开发者会立即收到通知

2. **持续部署(CD)**：
   ```
   测试通过 -> 自动部署到测试环境
   测试环境验证通过 -> 自动部署到预发布环境
   预发布环境验证通过 -> 自动部署到生产环境
   ```

### 这些脚本的使用方法

1. **code_security_audit.sh** (代码安全检查)
```bash
# 在提交代码前运行
./ci/code_security_audit.sh
```
- 这个脚本会：
  - 检查你修改的代码是否有安全隐患
  - 如果发现问题会阻止提交

2. **ci_local.sh** (本地测试)
```bash
# 在本地开发环境运行
./ci/ci_local.sh
```
- 这个脚本适合：
  - 开发新功能时快速测试
  - 只运行必要的测试用例
  - 不需要完整部署区块链节点

3. **ci_check.sh** (完整CI测试)
```bash
# 在CI服务器上运行
./ci/ci_check.sh
```
- 这个脚本会：
  - 部署完整的区块链环境
  - 运行所有测试用例
  - 检查所有功能

### 实际使用流程示例

1. 开发新功能时：
```bash
# 1. 写代码
vim your_code.py

# 2. 本地快速测试
./ci/ci_local.sh

# 3. 提交前安全检查
./ci/code_security_audit.sh

# 4. 提交代码
git add .
git commit -m "新功能：xxx"
git push
```

2. 代码提交后：
```
CI服务器自动运行 ci_check.sh
- 如果测试通过 -> 代码可以合并
- 如果测试失败 -> 需要修复问题
```

这就像是一条自动化的生产线：
- `ci_local.sh` 是你的工作台检查
- `code_security_audit.sh` 是质检员
- `ci_check.sh` 是最终的质量检验

这样可以：
- 保证代码质量
- 减少人工错误
- 加快开发效率
- 提高系统稳定性