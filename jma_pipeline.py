#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import kfp


# In[27]:


components_url = "/mnt/dkube/pipeline/components/"
dkube_training_op = kfp.components.load_component_from_file(
    components_url + "training/component.yaml"
)
dkube_serving_op = kfp.components.load_component_from_file(
    components_url + "serving/component.yaml"
)

token = os.getenv("DKUBE_USER_ACCESS_TOKEN")
client = kfp.Client(
    host=os.getenv("KF_PIPELINES_ENDPOINT"),
    existing_token=token,
    namespace=os.getenv("USERNAME"),
)
run_id = 0


# In[28]:


@kfp.dsl.pipeline(
    name='j_m_a',
    description='jungle mein aag prediction'
)

def jma_pipeline():

    # -------------------------------
    # TRAINING STEP
    # -------------------------------
    train = dkube_training_op(
        auth_token=token,
        container='{"image":"ocdr/d3-datascience-sklearn:v0.23.2-1"}',
        framework="sklearn",
        version="0.23.2",
        program="JMA-project",                  # Your DKube code repo
        run_script="python ff-train-model.py",        # Your training script
        datasets='["JMA-data"]',          # NAME OF DATASET IN DKUBE
        input_dataset_mounts='["/mnt/dataset"]',    # Will mount dataset at /data
        outputs='["j_m_a"]',             # Output model artifact
        output_mounts='["/mnt/result"]',          # Save model to /model
        envs='[]'
    )

    # -------------------------------
    # SERVING STEP
    # -------------------------------
    serving = dkube_serving_op(
        auth_token=token,
        model=train.outputs["artifact"],     # Model produced by training
        device="cpu",
        serving_image='{"image":"ocdr/sklearnserver:0.23.2"}',
#         transformer_image="",                 # No transformer needed
#         transformer_project="",               # empty
#         transformer_code=""                   # empty
    ).after(train)


# In[29]:


client.create_run_from_pipeline_func(jma_pipeline, arguments={})


# In[15]:
