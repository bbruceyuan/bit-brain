#!/bin/bash

download_dir="/DATA/disk2/yuhang/.cache/modelscope/datasets/swift/dolma"
# 定义文件 URL 列表
urls=(

  "https://olmo-data.org/dolma-v1_7/books/books-0000.json.gz"
  "https://olmo-data.org/dolma-v1_7/books/books-0001.json.gz"
  "https://olmo-data.org/dolma-v1_7/books/books-0002.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0000.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0001.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0002.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0003.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0004.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0005.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0035.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0036.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0037.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0038.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0039.json.gz"
  "https://olmo-data.org/dolma-v1_7/c4-filtered/c4-0040.json.gz"
)

# 循环遍历下载
for url in "${urls[@]}"
do
  echo "开始下载 $url 到 $download_dir" 
  wget -c "$url" -P $download_dir
done

echo "全部下载完成"
