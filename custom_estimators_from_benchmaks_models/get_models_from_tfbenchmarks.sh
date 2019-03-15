#!/bin/bash


# This scripts downloads the minimum necessary files from https://github.com/tensorflow/benchmarks
# in order to reuse the models implemented there.
# This has been tested so far for:
# - inception_model.py
# - resnet_model.py
# - vgg_model.py
# 
# The python script that import the models needs to be inside of `$workdir` defined bellow.
#
# In general the models can be defined like this:
# 
# from models.inception_model import Inceptionv3Model
# model = Inceptionv3Model().build_network((features, labels), phase_train=True, nclass=10).logits
# 
# from models.resnet_model import create_resnet101_model
# ResNet101Model = create_resnet101_model(None)
# model = ResNet101Model().build_network((features, labels), phase_train=True, nclass=10).logits
#
# from models.vgg_model import Vgg16Model
# model = Vgg16Model().build_network((features, labels), phase_train=True, nclass=10).logits

tag="cnn_tf_v1.12_compatible"
base_url="https://raw.githubusercontent.com/tensorflow/benchmarks/${tag}/scripts/tf_cnn_benchmarks"
workdir="models_from_benchmark"

# Add here the model to download
list_of_models=`cat << EOF
inception_model.py
resnet_model.py
vgg_model.py
EOF
`

# Skip if the folde exists
if [ -d "${workdir}" ]; then
	echo "The models are already downloaded!"
	echo "Please, delete ${workdir} to redownload them."
	exit
else
	mkdir -p ${workdir}/models
fi

# Download the models
for model in $list_of_models; do
	wget ${base_url}/models/$model -O ${workdir}/models/$model
done

# Download utilitiy files
wget  ${base_url}/models/model.py -O ${workdir}/models/model.py
wget  ${base_url}/convnet_builder.py -O ${workdir}/convnet_builder.py

# Create the `datasets.py` file. This file exists also on the repo at the same
# level as `convnet_builder.py` but so far it is only needed by `resnet_model.py`
# just for the definition of `IMAGENET_NUM_TRAIN_IMAGES`. To avoid unnecessary
# files from the repo, the file will be created with the required definition.
# If later some model needs much more things from it, then it will have to be
# downloaded.
datasets=`cat << EOF
IMAGENET_NUM_TRAIN_IMAGES = 1281167
EOF`

echo $datasets > ${workdir}/datasets.py

patch_convnet_builder=`cat << EOF
diff -Nru models_from_benchmark/convnet_builder.py models_from_benchmark.edited/convnet_builder.py
--- models_from_benchmark/convnet_builder.py	2019-02-20 18:16:28.000000000 +0100
+++ models_from_benchmark.edited/convnet_builder.py	2019-02-20 18:17:26.000000000 +0100
@@ -206,7 +206,8 @@
                                      initializer=tf.constant_initializer(bias))
           biased = tf.reshape(
               tf.nn.bias_add(conv, biases, data_format=self.data_format),
-              conv.get_shape())
+              #conv.get_shape())
+              tf.shape(conv))
         else:
           biased = conv
       else:
EOF
`
echo "$patch_convnet_builder" > patch_convnet_builder.patch
patch -d $workdir -p1 < patch_convnet_builder.patch
rm patch_convnet_builder.patch
