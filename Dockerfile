FROM mambaorg/micromamba:1.5.8
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes
WORKDIR /workspace
COPY . /workspace
ENV PATH="/opt/conda/bin:${PATH}"
CMD ["bash", "scripts/run_demo.sh"]
