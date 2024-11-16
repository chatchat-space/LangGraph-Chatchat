# 基础镜像
FROM python:3.12
LABEL maintainer=yuehua-s
WORKDIR /root

# 环境变量
ENV CHATCHAT_ROOT=/root/chatchat_data

# 初始化环境
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone
# 初始化环境
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get update -y && \
    apt-get install -y git && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    apt-get install -y default-mysql-client default-mysql-server && \
    apt-get install -y libpq-dev python3-dev postgresql postgresql-contrib && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools
RUN pip install --index-url https://pypi.python.org/simple/ pipx && \
    pipx install poetry --force

# Add poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# 添加 SSH 密钥(项目未公开前临时使用)
# 假设您已经将 SSH 密钥添加到 Docker build 的上下文中
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
# 添加 GitHub 到已知主机
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# 下载 LangGraph-Chatchat
RUN #git clone https://github.com/chatchat-space/LangGraph-Chatchat.git
RUN git clone git@github.com:chatchat-space/LangGraph-Chatchat.git

# 安装依赖
WORKDIR /root/LangGraph-Chatchat/chatchat-server
RUN poetry config virtualenvs.create false
RUN poetry install --with lint,test -E xinference

## 确保 Python 可以找到 chatchat 模块
ENV PYTHONPATH="/root/LangGraph-Chatchat/chatchat-server:${PYTHONPATH}"

# 初始化配置
WORKDIR /root/LangGraph-Chatchat/chatchat-server/chatchat
RUN python cli.py init

## 初始化知识库文件
#ADD chatchat_data.tar.gz $CHATCHAT_ROOT/

EXPOSE 7861 8501
ENTRYPOINT ["python", "cli.py", "start", "-a"]