# prod version
# docker run -dit --gpus all --shm-size=64g --name pcos-roi-classification -v $(pwd)/:/workspace/ pcos-roi-classification:latest

# dev version
docker run -dit --gpus all --shm-size=64g --name pcos-roi-classification \
  -v $(pwd)/:/workspace/ \
  -v /mnt/sdd/pcos_dataset/:/workspace/pcos_dataset \
  kor_med_qa_benchmark:latest