#!/bin/bash -e

export DOCKER_BUILDKIT=1

docker build -t docker.flatironinstitute.org/mcranmer/srbench-core:latest .
# Bash array of what to push to the registry:
to_push=( "docker.flatironinstitute.org/mcranmer/srbench-core" )

old_dir=$(pwd)
base_dir="/mnt/home/mcranmer/pysr_paper_syw/srbench-comp/official_competitors"
cd $base_dir
for dir in *; do
    lower_dir=$(echo $dir | tr '[:upper:]' '[:lower:]')
    cd $base_dir/$dir
    docker build -t docker.flatironinstitute.org/mcranmer/$lower_dir:latest .
    to_push+=( "docker.flatironinstitute.org/mcranmer/$lower_dir" )
done

cd $old_dir

echo "Finished build. Now, updating images."
for image in "${to_push[@]}"; do
    nohup docker push $image &
done

# VERSION=$1
# # assert version is not empty:
# if [ -z "$VERSION" ]; then
#     echo "Version is empty"
#     exit 1
# fi
# DOCKER_BUILDKIT=1 docker build -t docker.flatironinstitute.org/mcranmer/pysr-srbench:latest .
# docker image tag docker.flatironinstitute.org/mcranmer/pysr-srbench:latest docker.flatironinstitute.org/mcranmer/pysr-srbench:$VERSION
# docker push docker.flatironinstitute.org/mcranmer/pysr-srbench