from pathlib import Path

for dir in Path("official_competitors").glob("*"):
    basename = dir.name
    with open(dir / "Dockerfile", "w") as f:
        print(f"FROM docker.flatironinstitute.org/mcranmer/srbench-core:latest", file=f)
        print(f"# Install {basename}", file=f)
        print(f"COPY . /tmp/{basename}", file=f)
        print(f"WORKDIR /tmp/{basename}", file=f)
        # Check if any .zip file in this folder:
        if len(list(dir.glob("*.zip"))):
            print("RUN unzip *.zip", file=f)
        print(f"RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -n method -f environment.yml", file=f)
        print(f"RUN mamba run -n method /bin/bash install.sh", file=f)
        print("", file=f)