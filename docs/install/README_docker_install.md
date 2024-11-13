# LangGraph-Chatchat 容器化部署指引

> 提示: 此指引为在 Linux(CentOS) 环境下编写完成, 其他环境下暂未测试, 理论上可行.
> 
> LangGraph-Chatchat docker 镜像已支持多架构(amd64/arm64).

## 一. LangGraph-Chatchat 部署

### 1. 安装 docker 和 docker-compose
- 安装 docker 文档参见: [Install Docker Engine](https://docs.docker.com/engine/install/).

- 安装 docker-compose 文档参见:
- - [Install Compose standalone](https://docs.docker.com/compose/install/standalone/) 
- - [版本列表](https://github.com/docker/compose/releases)
- - 举例: Linux X86 环境 可下载 [docker-compose-linux-x86_64](https://github.com/docker/compose/releases/download/v2.27.3/docker-compose-linux-x86_64) 使用.
```shell
cd ~
wget https://github.com/docker/compose/releases/download/v2.27.3/docker-compose-linux-x86_64
mv docker-compose-linux-x86_64 /usr/bin/docker-compose
chmod +x /usr/bin/docker-compose
which docker-compose
```
```text
/usr/bin/docker-compose
```
```shell
docker-compose -v
```
```text
Docker Compose version v2.27.3
```

### 3. 下载 LangGraph-Chatchat
```shell
cd ~
git clone https://github.com/chatchat-space/LangGraph-Chatchat.git
cd LangGraph-Chatchat/docker/
tar -xvf chatchat_data.tar.gz -C /root/
cp docker-compose.yaml /root/docker-compose.yaml
```

### 4. 配置模型
修改`model_settings.yaml`中`DEFAULT_LLM_MODEL`,`DEFAULT_EMBEDDING_MODEL`,`MODEL_PLATFORMS`,`api_key`等配置.
```shell
[root@VM-1-10-centos chatchat_data]# ls
basic_settings.yaml  data  kb_settings.yaml  model_settings.yaml  prompt_settings.yaml  tool_settings.yaml
[root@VM-1-10-centos chatchat_data]# ll
total 28
-rw-r--r-- 1 root root 2191 Nov 12 21:21 basic_settings.yaml
drwxr-xr-x 6 root root 4096 Nov 13 01:49 data
-rw-r--r-- 1 root root 3382 Nov 12 21:21 kb_settings.yaml
-rw-r--r-- 1 root root 3879 Nov 13 00:53 model_settings.yaml
-rw-r--r-- 1 root root 5415 Nov 12 21:21 prompt_settings.yaml
-rw-r--r-- 1 root root 3641 Nov 13 00:55 tool_settings.yaml
```

### 5. 启动服务
```shell
cd ~
docker-compose up -d
```
```text
WARN[0000] /root/docker-compose.yaml: `version` is obsolete 
[+] Running 2/2
 ✔ Container root-chatchat-1    Started
```
```shell 
 docker-compose ps      
```

### 6. 查看服务日志
```shell
docker-compose logs -f chatchat
```

### 7. 打开浏览器
浏览器访问 http://<你机器的ip>:8501

## 二. 本地模型推理部署(可选)
### 1. 安装 NVIDIA Container Toolkit
寻找适合你环境的 NVIDIA Container Toolkit 版本, 请参考: [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

安装完成后记得按照刚刚文档中`Configuring Docker`章节对 docker 进行初始化.

### 2.部署 LLM 推理服务服务

#### Xinference
参见 [Xinference Docker Image](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html)

#### Ollama
参见 [Download Ollama](https://ollama.com/download)