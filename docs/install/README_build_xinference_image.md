本文指导如何打一个 Xinference[v1.0.0]启动模型的 docker 镜像
1. 参考文档 [自定义镜像](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_docker_image.html#dockerfile-for-custom-build)
2. 下载 Xinference 默认镜像
3. 启动 Xinference 默认镜像, 并将系统文件 /data 挂载进容器中
```shell
docker run -itd \
  -v ~/data/.xinference:/root/.xinference \
  -v ~/data/.cache/huggingface:/root/.cache/huggingface \
  -v ~/data/.cache/modelscope:/root/.cache/modelscope \
  -p 9997:9997 \
  --gpus all \
  test:latest \
  xinference-local -H 0.0.0.0
```
4. 启动后, 登陆 UI 下载需要的模型
确认
```shell
(base) [root@VM-128-14-tencentos ~]# cd data
(base) [root@VM-128-14-tencentos data]# ls
(base) [root@VM-128-14-tencentos data]# ls -la
total 16
drwxr-xr-x  4 root root 4096 Nov 16 23:00 .
dr-xr-x--- 11 root root 4096 Nov 17 16:06 ..
drwxr-xr-x  4 root root 4096 Nov 16 23:00 .cache
drwxr-xr-x  4 root root 4096 Nov 16 23:14 .xinference
(base) [root@VM-128-14-tencentos data]# du -sh .cache
17G     .cache
(base) [root@VM-128-14-tencentos data]# du -sh .xinference
216K    .xinference
```
5. 将服务停掉, 开启打镜像
```shell
git clone https://github.com/xorbitsai/inference.git
cd inference
cp ~/data .
```
预期
```shell
(base) [root@VM-128-14-tencentos ~]# cd inference/
(base) [root@VM-128-14-tencentos inference]# ls
assets     data  examples  MANIFEST.in     README_ja_JP.md  README_zh_CN.md  setup.py       xinference
benchmark  doc   LICENSE   pyproject.toml  README.md        setup.cfg        versioneer.py
(base) [root@VM-128-14-tencentos inference]# du -sh data/.cache
17G     data/.cache
(base) [root@VM-128-14-tencentos inference]# du -sh data/.xinference/
212K    data/.xinference/
```
6.修改 Dockerfile
```shell
(base) [root@VM-128-14-tencentos ~]# cd inference/
(base) [root@VM-128-14-tencentos inference]# vim xinference/deploy/docker/Dockerfile
```
在 Dockerfile 最后新增如下内容
```text
# copy model
COPY data /root/
```
整体内容(v1.0.0)如下：
```text
FROM vllm/vllm-openai:v0.6.0

COPY . /opt/inference
WORKDIR /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y curl procps git libgl1 ffmpeg \
  # upgrade libstdc++ and libc for llama-cpp-python
  && printf "\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
  && apt-get -y update \
  && apt-get install -y --only-upgrade libstdc++6 && apt install -y libc6 \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib

ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --upgrade -i "$PIP_INDEX" pip && \
    pip install -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3" && \
    # use pre-built whl package for llama-cpp-python, otherwise may core dump when init llama in some envs
    pip install "llama-cpp-python>=0.2.82" -i https://abetlen.github.io/llama-cpp-python/whl/cu124 && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements.txt && \
    pip install -i "$PIP_INDEX" --no-deps sglang && \
    pip uninstall flashinfer -y && \
    pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4 && \
    cd /opt/inference && \
    python3 setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    # clean packages
    pip cache purge

# copy model
COPY data /root/

# Overwrite the entrypoint of vllm's base image
ENTRYPOINT []
CMD ["/bin/bash"]
```
7. 打镜像
```shell
docker build --progress=plain -t chatchatspace/xinference:qwen2.5 -f xinference/deploy/docker/Dockerfile --platform=linux/amd64 . 
```
8. 启动容器
```shell
docker run -itd -p 9997:9997 --gpus all chatchatspace/xinference:qwen2.5 xinference-local -H 0.0.0.0 --log-level debug
```