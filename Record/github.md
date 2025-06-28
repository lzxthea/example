# github与vscode配置

## github下载与克隆仓库

1. 在 GitHub 上创建新仓库
   1. 访问 [GitHub](https://github.com/?lang=zh-CN&open_in_browser=true) 并登录账号。
   2. 点击右上角的 + 号，选择 New repository。
   3. 填写仓库信息：
      1. Repository name：输入仓库名称（如 my-project）。
      2. Description：可选，简要描述仓库内容。
      3. Public/Private：选择仓库可见性。
      4. Initialize this repository with：勾选 Add a README file（可选，但推荐）。
      5. 可添加 .gitignore（如选择 Python 模板）和 License。
   4. 点击 Create repository。
2. 在本地克隆仓库
   1. 在本地安装git。可以在终端中输入 git 来检查是否已经安装，如果未安装，**下载地址：**[git-scm.com](https://git-scm.com/?lang=zh-CN&open_in_browser=true)
   2. 生成本地ssh密钥
      1. 打开终端，执行命令来查看是否已经存在 SSH 密钥。
         ```
         ls -al ~/.ssh
         ```
         如果输出中包含 id_rsa.pub、id_ed25519.pub 这样的文件，说明已经存在 SSH 密钥；如果没有，则需要生成新的 SSH 密钥。
      2. 如果没有现有的 SSH 密钥，生成一个新的 SSH 密钥对：
         ```
         ssh-keygen -t ed25519 -C "your_email@example.com"
         ```
         将 your_email@example.com 替换为你在 GitHub 上注册的邮箱地址。在生成过程中，会提示你选择保存密钥的位置和设置密码，建议一路回车使用默认值。
      3. 将 SSH 密钥添加到 SSH 代理
         ```
         eval "$(ssh-agent -s)"
         ssh-add ~/.ssh/id_ed25519
         ```
      4. 将公钥添加到 GitHub 账户
         ```
         eval "$(ssh-agent -s)"
         ssh-add ~/.ssh/id_ed25519
         ```
         将输出的内容复制下来，然后登录 GitHub 账户，进入 Settings -> SSH and GPG keys -> New SSH key，将复制的公钥粘贴到 Key 字段中，给密钥设置一个合适的标题，最后点击 Add SSH key。
      5. 测试 SSH 连接
         ```
         ssh -T git@github.com
         ```
         如果看到类似以下的输出，说明连接成功：
         ```
         Hi username! You've successfully authenticated, but GitHub does not provide shell access.
         ```
   3. 克隆仓库
        ```
        git clone 你复制的仓库ssh
        ```
3. 克隆其他人的仓库
   1. 克隆你的仓库到本地
        ```
        git clone https://github.com/yourusername/yourrepository.git
        cd yourrepository
        ```
   2. 从别人的仓库复制所需文件
