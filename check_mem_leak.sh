#!/bin/bash

CUDA_VISIBLE_DEVICES=0 AB_REQUEST_TYPE=ADSBRAIN_BOND AB_ENTRYPOINT=/v2/models/jim/versions/1/infer /usr/local/bin/valgrind \
	--leak-check=yes \
	--suppressions=valgrind_suppression.txt \
	/datadrive/fhu/github/triton-server-abo-v2/triton-server/opt/tritonserver/bin/tritonserver --model-repository /datadrive/fhu/github/abo-models/team/selection/mistv3/m/ --allow-adsbrain true --adsbrain-port 8888

