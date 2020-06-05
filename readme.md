# 1. Linux 环境和重装

## 1.1 系统安装

popos：custom 安装/ /boot swap 不用安装/home 

## 1.2 启动表修复

grub-repair 修复启动表 进入linux

gurb customizer 修复win启动表

## 1.3 挂载home盘

自动挂载原home盘 runtime覆盖原有根目录下自带home盘

```shell
vim /etc/fstab 
# 添加如下行
/dev/sda5  /home  ext4  defaults  0  0 
```

## 1.4 应用修复 zsh修复

```shell
apt install vim tmux ranger zsh flameshot ntpdate make
chsh -s /bin/zsh #输入密码 重启 修复zsh
```

安装chrome vscode 搜狗输入法 卸载ibus 安装fcitx

## 1.5 统一双系统时间

```shell
sudo ntpdate time.windows.com 
sudo hwclock --localtime --systohc 
```


## 1.6 设置免密码sudo     

```shell
vim /etc/sudoers  
# 更改用户组的nopasswd 除了root全部改称nopasswd格式
%admin ALL=(ALL) NOPASSWD: ALL
```

## 1.7 安装邮箱

mailspring

qq邮箱验证码xwsxzxowqgupbddc当作密码

# 2. GCC

gcc 7.3 

## 2.1 源代码make 或者直接apt install

```shell
wget https://ftp.gnu.org/gnu/gcc/gcc???  # 下载镜像源
sudo make 
sudo make check
sudo make install  #一定要sudo 否则奇怪问题 
```

install 的 bin文件在/usr/local/bin/gcc   并不在/usr/bin（添加源之后apt install 的gcc在这个路径）

## 2.2 多版本并存切换

```sh
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 40 
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc-7 60
sudo update-alternatives --config gcc  # 控制默认gcc版本 g++同上
```

# 3. Anaconda虚拟环境
## 3.1 安装

sh文件装好 

.zshrc  或者.bashrc注册  source刷新
 注意添加sock5的支持 与本机代理同步：

```shell
unset all_proxy && unset ALL_PROXY
pip install pysocks  #添加sock支持
```

## 3.2 设置虚拟环境

```shell
create -n name python=3.6   #3.6对应了tf1.7版本 
pip install --upgrade pip
```

## 3.3 初次使用添加bazel构建的支持

新建立的conda虚拟环境中 预先安装对应的库

```shell
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
pip install h5py==2.8.0
```

注意：尽量不要再base环境安装tensorflow和jupyter notebook之类的包，防止虚拟环境没有安装而去上级base环境找到直接使用引起版本错误

# 4. TensorFlow安装和构建

注意：pip方式安装容易引起和本地custom op的补丁版本不一致 建议本地同版本build成whl后安装

对齐GCC和bazel版本

## 4.1 bazel

安装

https://github.com/bazelbuild/bazel/releases  下载对应版本的installer-linux-x86_64.sh

```shell
chmod +x bazelxxxxx.sh 
./bazelxxxxxx.sh --user
```

卸载

```shell
rm -rf ~/.bazel
rm -rf ~/bin
rm -rf /usr/bin/bazel    
```

一般只是切换不需要更改.zshrc文件 先卸载再安装后 用bazel version验证版本

## 4.2 TensorFlow源码的编译配置configure

### 4.2.1 ./configure先自动配置生成配置文件

第一步询问的python解释器地址 是conda env的解释器 所以需要先启动虚拟环境

clang可以选y    XLA可以选false  cuda为GPU支持  其他都是默认N

workspace 在build安卓aar时使用 本地tensorflow和lite不需要设置 默认N即可

### 4.2.2 vim .tf_configure.bazelrc 再手动修改配置文件

参考GCC手册：对于cpu优化标记  https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html

更改  build:opt --copt=-march=native（本地支持）   为   build:opt --copt=-mtune=generic（通用支持）

更改  build:opt --host_copt=-march=native   为   build:opt --host_copt=-mtune=generic

## 4.3 build相关

tensorflow目录下的bazel-bin等目录是.cache的链接

### 4.3.1 构建tensorflow

```shell
bazel build //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"   #构建pip打包器 ABI参数为gcc5以上 兼容旧版本 单独环境可以无视

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg   #打为whl包

source activate xxx
pip install /tmp/tensorflow_pkg/tensorflow-cp36xxxx.whl    #安装tensorflow
```

### 4.3.2 单独构建tensorflow custom op的.so共享库

tensorflow/core/user_ops目录下

添加新算子的.cc源文件 格式参见tf官网

```shell
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print("".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print("".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zerof.cc -o zerof.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0 -O2
```

g++构建为 zerof.so文件 使用bazel会出现google protobuf依赖找不到等诡异问题

```python
tf.load_op_library('??/zerof.so').zerof()  #调用so中的op
```

### 4.3.3 单独构建python tensorflow lite解释器的.so共享库

注意：

​    仅在tf2.1下实验成功 

​    之前版本使用SWIG进行cpp和python之间的交互   后续tf官方将改为pybind11的方式实现交互

​    单独更新该库 即可实现tensorflow lite之中的算子  包括：增加custom的op 以及 覆盖builtin的op   无需整体编译tf whl

```shell
bazel build --config opt //tensorflow/lite/python/interpreter_wrapper:tensorflow_wrap_interpreter_wrapper
```

构建结果为/bazel-bin/tensorflow/lite/python/interpreter_wrapper/_tensorflow_wrap_interpreter_wrapper.so

替换到~/anaconda3/envs/tf20/lit/python3.6/site-packages/tensorflow_core/lite/python/interpreter_wrapper/ 路径下

注意：对于tf 2.1 包路径是tensorflow_core 而不是之前的tensorflow

#### 增添custom op

##### a. 编写算子源文件 

格式为：

```c++
// zero.cc
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {    // 注意1.x版本官方是放到了builtin命名空间 2.1需要放在custom命名空间
namespace zerof {
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {}                             
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {}
}  // namespace zerof
TfLiteRegistration* Register_ZEROF() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 zerof::Prepare, zerof::Eval};
  return &r;                     
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

在kernels路径下

##### b. BUILD文件添加规则

```python
cc_library(
    name = "builtin_op_kernels",
    srcs = [
        "activations.cc",
        "add.cc",
        "add_n.cc",
        "arg_min_max.cc",
....
         "zerof.cc",   ## 添加在这里 打包所有的算子
    ],
    hdrs = [
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android() + EXTRA_EIGEN_COPTS,
```

##### c. 添加声明

在 (builtin_opkernels.h 后经验证不用)   custom_ops_register.h 添加

```c++
TfLiteRegistration* Register_ZEROS();
```

##### d. 注册算子

在 register.cc   register_ref.cc 添加

```c++
namespace custom {
TfLiteRegistration* Register_ZEROF();
}

BuiltinOpResolver::BuiltinOpResolver(){
AddCustom("Zerof", tflite::ops::custom::Register_ZEROF());  
    // Zerof 这里我们需严格使用大驼峰命名 例如: ExtractImagesPatches
    // 在编译为解释器之后 算子名将被自动转化为下划线所谓snack格式 例如: extract_images_patches
    // 另注: Register_ZEROF这个名称没有上面两条的限制 随便定义 统一即可
}
```

##### e. polish编译验证流程

使用bazel build一次之后，tensorflow的源代码文件夹下已经包含了解释器的so文件

跳转到conda虚拟环境的tensorflow代码位置：~/anaconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/lite/python/interpreter_wrapper

（注意：tf2.x以上在...site-packages/tensorflow_core/...    tf1.x则是在...site-packages/tensorflow/..）

为解释器so文件建立软连接

```shell
ln -s /home/gx/myproj/tensorflow/bazel-bin/tensorflow/lite/python/interpreter_wrapper/_tensorflow_wrap_interpreter_wrapper.so _tensorflow_wrap_interpreter_wrapper.so
```

每次更改算子kernel源代码 -> bazel build -> restart解释环境 例如jupyter notebook 即可使用更新的op算子

##### f. 算子增加属性

在tflite中custom op的属性和tf中custom op的属性不同，是因为保存模型的格式不同

tflite使用flexbuffer进行模型保存 是flatbuffer的一个精简子集

以extract_image_patches算子为例，tf中该算子的属性包括

```c++
 std::vector<int32> ksizes_;
 std::vector<int32> strides_;
 std::vector<int32> rates_;
 string padding_;
```

为增加四个属性，需要在custom op的kernel源代码中实现Init和Free方法

首先，在命名空间中定义一个结构体用来暂存各个属性

```c++
namespace extract_image_patches {
typedef struct {
  std::vector<int32> ksizes_;
  std::vector<int32> strides_;
  std::vector<int32> rates_;
  string padding_;
} TfLiteEIPOParams;
...
```

第二，在Init方法中从tflite的flexbuffer格式模型中读出属性值，存入上述结构体

```c++
// 加上相关include
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include <vector>
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/op_macros.h"


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new TfLiteEIPOParams;//开辟一块空间存属性值
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);//属性存在模型的buffer里面
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();//参考flexbuffer官网文档 m可以理解为一个json
  
  //这里需要用AsTypedVector读出，如果用AsVector会失败，原因不明，可能flexbuffer做了奇怪的优化
  //注意：此处读出为vector其实不是cpp的std vector 而是flexbuffer的vector 
  //     所以还需通过AsInt32等方式获取为数值
  //     同样的AsString()之后需要用c_str()转化为cpp的格式
  data->rates_.push_back(m["rates"].AsTypedVector()[0].AsInt32());
  data->rates_.push_back(m["rates"].AsTypedVector()[1].AsInt32());
  data->rates_.push_back(m["rates"].AsTypedVector()[2].AsInt32());
  data->rates_.push_back(m["rates"].AsTypedVector()[3].AsInt32());

  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[0].AsInt32());
  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[1].AsInt32());
  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[2].AsInt32());
  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[3].AsInt32());

  data->strides_.push_back(m["strides"].AsTypedVector()[0].AsInt32());
  data->strides_.push_back(m["strides"].AsTypedVector()[1].AsInt32());
  data->strides_.push_back(m["strides"].AsTypedVector()[2].AsInt32());
  data->strides_.push_back(m["strides"].AsTypedVector()[3].AsInt32());
    
  data->padding_ = m["padding"].AsString().c_str();

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteEIPOParams*>(buffer);//最后释放空间
}
```

第三，在Prepare和Eval方法中，查寻该属性

```c++
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
...
auto* params = reinterpret_cast<TfLiteEIPOParams*>(node->user_data);
//其实属性被解释器读取后 放在node的user_data之中 读出并用TfLiteEIPOParams格式化
auto temp = params->ksizes_[0]; //使用属性值
...
}
```



### 4.3.4 构建Android解释器aar包

jitbrain Android Studio(AS)（自带安卓skd ） +  ndk18b（验证可用 网上说必须15 垃圾）  +    bazel    tf2.1

配置tf Download a fresh release of clang : y   自配workspace 指定上面安好的 sdk ndk 路径

必要时修改配置文件.tf_configure.bazelrc： build --action_env ANDROID_NDK_API_LEVEL = 21 

```shell
bazel build --cxxopt='--std=c++11' -c opt        \
  --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a   \    # 两个arm基本包含大部分安卓手机
  //tensorflow/lite/java:tensorflow-lite
```

构建 .aar文件 结果保存在

bazel-genfiles/tensorflow/contrib/lite/java/tensorflow-lite.aar

在AS导入并在build gradle里写入依赖：implementation project(':tensorflow-lite')

#### 增添custom op

##### a. 编写算子源文件

直接将算子源文件 改扩展名为zerof.h  并放置在tensorflow/lite/java/src/main/native路径下

##### b. BUILD文件添加规则

```python
# *This includes all ops. If you want a smaller binary, you should copy and*
# *modify builtin_ops_jni.cc.  You should then link your binary against both*
# *":native_framework_only" and your own version of ":native_builtin_ops".*
# 如果要精简op集 进一步的减小tflite模型大小 参见下文c部分
cc_library(
    name = "native",
    srcs = [
        "builtin_ops_jni.cc",
    ],
    hdrs = [
        "zerof.h",    ## 头文件的build说明添加在这里 给builtin_op_jni.cc添加该算子
    ],
    copts = tflite_copts(),
    deps = [
        ":native_framework_only",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)
```

##### c. 注册算子

在builtin_op_jni.cc 文件内添加头文件声明

```c++
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/java/src/main/native/zerof.h"    #改成头文件 添加在这里
namespace tflite {
...
```

##### d. 精简op集

register.h 的 BuiltinOpResolver 类的构造方法  官方已经默认在register.cc中实现  

为了精简op集 可以自己overload该方法 

builtin_op_jni.cc 原始代码：

```c++
std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
      new tflite::ops::builtin::BuiltinOpResolver());
}
```

在原本的return里面 替换为：

```c++

std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
    new tflite::ops::builtin::BuiltinOpResolver::BuiltinOpResolver() {
      AddBuiltin(BuiltinOperator_ABS, Register_ABS());
      AddBuiltin(BuiltinOperator_MATRIX_SET_DIAG, Register_MATRIX_SET_DIAG());
    ... //删减一些算子
      AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
      AddCustom("Zerof", tflite::ops::custom::Register_ZEROF());
    } // end of BuiltinOpResolver(){}
  );  // end of return
} 
```

# 5. TensorFlow模型格式及转换

frozen固化模型是为了移动端部署实现的 但是lite也实现了该功能 

推荐方法简单处理后 将模型保存为 saved_model 并转化为lite模型

在tf2.x环境直接支持tf1.x模型运行：

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```



## 5.1 处理模型 保存为saved_model

```python
input_image_holder = tf.placeholder(tf.float32, [1,512,1360,3], name='input_image_holder') 
#设置输入占位符 类型 维度 node名称 
...
tf.saved_model.simple_save(sess,
	"./save_model_path",  #输出路径
	inputs={"input_image_holder": input_image_holder},  # node名 输入对象 
	outputs={"saturate_cast": output}) # 输出node名（根据实际情况） 输出对象
```

## 5.2 将saved_model 转化为lite模型

给lite添加custom op 分两种情况：

1. tensorflow含有该算子 而lite没有  

   ​		可以参照官网 使用tf算子库 后果是模型变得很大 官方还有白名单 有些op并不支持添加

   ​		可以自己在lite中实现该op  推荐

2. tensorflow不包含该算子 想在lite之中添加

   ​        需要先在官方tf库中 利用.so文件添加该算子  接着在lite的python解释器中添加  最后在移动端解释器添加

```python
tf.load_op_library('??/zerof.so')  ## 必须先导入custom op 否则无法识别 
converter = tf.lite.TFLiteConverter.from_saved_model('./save_model_path_zerof') # 导入
converter.allow_custom_ops = True  ## 事实上这里是：允许tf包含 但是lite不包含的op被打包进lite模型
# converter.post_training_quantize = True   ## 模型后处理量化开关
tflite_model = converter.convert()
open("./1.tflite", "wb").write(tflite_model)
```

## 5.3 pc机python环境导入并运行lite模型

```python
interpreter = tf.lite.Interpreter(model_path='/home/gx/myproj/generative_inpainting-master/1.tflite')  
# 导入模型创建解释器 如果custom op在kernels目录的某处没有注册 会提示不认识的符号
interpreter.allocate_tensors()  # 分配张量
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

input_shape = input_details[0]['shape'] 
input_data = np.array(np.random.random_sample(input_shape), dtype=np.int32) # 随机输入
print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index']) # 取输出观察
print(output_data)
```









