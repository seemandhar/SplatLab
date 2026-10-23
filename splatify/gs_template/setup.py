from setuptools import setup

setup(
    name="gs-template",
    version="0.1.0",
    packages=["gs_template"],
    entry_points={
        "nerfstudio.plugins": [
            "gs-template = gs_template.configs:gs_template_method",
            "absgs = gs_template.configs:absgs_method",
            "depth-reg-gs = gs_template.configs:depth_reg_gs_method",
            "feature-gs = gs_template.configs:feature_gs_method",
            "micro-splatting = gs_template.configs:micro_splatting_method",
            "bubble-gs = gs_template.configs:bubble_gs_method",
        ],
    },
)
