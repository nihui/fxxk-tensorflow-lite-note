# 0 安装准备

## 0.1 protobuf



下载 解压 https://github.com/protocolbuffers/protobuf/releases

```shell
$ ./autogen.sh     ##下载自github的代码需要执行此行来生成configure文件
$ ./configure
$ make
$ make check
$ make install

# protobuf会被安装到usr/local/bin，usr/local/lib，usr/local/include
```

```shell
$ export LD_LIBRARY_PATH=/usr/local/lib  # 最好加入~/.zshrc
```

```shell
$ protoc --version   # 验证
```

## 0.2 opencv

### 安装

依赖项

```shell
sudo apt-get install build-essential libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

下载解压

```shell
mkdir build
cd ./build
```

使用cmake生成make

```shell
cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=ON -D CMAKE_INSTALL_PREFIX=/usr/local ..
```

OpenCV默认不生成.pc文件，OPENCV_GENERATE_PKGCONFIG=ON才会生成。该编译选项开启生成opencv4.pc文件，支持pkg-config功能。
可用-D CMAKE_INSTALL_PREFIX指定安装目录。如果指定安装目录则默认各部分分别安装在/usr/local/目录的include/ bin/ lib/3个文件夹下.

```shell
sudo make -j4
sudo make install
cd /etc/ld.so.conf.d/  # 配置
sudo touch opencv.conf
sudo sh -c 'echo "/usr/local/lib" > opencv.conf'
sudo ldconfig # 更新pkg-config
pkg-config --libs opencv4  # 验证
pkg-config --cflags opencv4
```

### 卸载

```shell
sudo rm /etc/ld.so.conf.d/opencv.conf
cd ./OpenCV-xxx/build
sudo make uinstall
```



## 0.3 cmake

下载tar.gx版本解压到home

```shell
export PATH=$PATH:/home/gx/cmake-3.17.2-Linux-x86_64/bin    # 最好加入~/.zshrc
```

## 0.4 ncnn

clone源代码

对于examples/CMakeLists.txt，首行加入

```cmake
set(OpenCV_DIR /home/gx/myproj/opencv-3.4.9/build)
```

新建build文件夹在其下输入

```shell
cmake -D NCNN_BUILD_EXAMPLES=true ..     # 打开example的编译
make -j4   # 出奇怪问题 检查protoc --version
```



# 1 输入图片

ncnn中往往不支持不同通道数的mat自动运算，例如mask在tf中可以一个通道输入，在ncnn中还需要补全到三通道

需注意ncnn中的运算带有广播功能

```c++
cv::Mat image = cv::imread("/home/gx/myproj/tensorflow2ncnn/build/examples/case1_input.png", 1); //导入为opencv的mat类型

cv::Mat mask = cv::imread("/home/gx/myproj/tensorflow2ncnn/build/examples/case1_mask.png",1);//导入mask
cv::Mat mask_b = mask.clone();
cv::threshold(mask,mask_b,127.5f,1.0f,CV_THRESH_BINARY);//二值化mask 大于127.5取1.0
```



# 2 导入网络

ncnn的mat类型类似与opencv的mat类型，可以通过from_pixels和to_pixels进行二者的转换 

特别的：opencv的通道为BGR排布，但是导入ncnn时使用 PIXEL_RGB也无所谓，输出同样用PIXEL_RGB保证顺序不变即可

```c++
ncnn::Net mynet;

mynet.load_param("mynet_test.param");
mynet.load_model("mynet_test.bin");

// input image
//使用from_pixels方法 将opencv的mat转换为ncnn的mat
ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows); 

const float mean_vals[3] = {127.5f,127.5f,127.5f};
const float normalize_vals[3] = {1/127.5f,1/127.5f,1/127.5f};
in.substract_mean_normalize(mean_vals, normalize_vals); // 归一化0-255 图片到-1.0 +1.0

// input mask
ncnn::Mat mask = ncnn::Mat::from_pixels(bgr_mask.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows);

```



Mat之间赋值应使用指针逐个更改，不可以直接用inpaint.channel(0) = temp.channel(0) 进行通道赋值 

如果用上述方法进行通道赋值，则inpaint.channel 0 并不会真的改变，反而会产生野指针指向其他数据

而且直接用mat赋值可以，单独channel互相赋值就不行

```c++
// inpaint_net input
ncnn::Mat inpaint(bgr.cols, bgr.rows, 5, 4u);   //构建一个5通道的mat 012通道为原始图片*（1-mask） 3为全1 4为mask的0通道
ncnn::Mat temp(bgr.cols, bgr.rows, 1, 4u);
temp.fill(1.0f);

//处理 0 1 2通道
for (int q=0; q<3; q++){
    float* ptr = in.channel(q);  // mat.channel返回一个新mat的指针
    float* ptd = inpaint.channel(q);
    float* ptm = mask.channel(0);
    for (int y=0; y<in.h; y++){
        for (int x=0; x<in.w; x++){
            ptd[x] = ptr[x]*(1-ptm[x]);
        }
        ptd += in.w;
        ptr += in.w;
        ptm += in.w;
    }
}
//处理3通道
float* ptr3 = temp.channel(0);
float* ptd3 = inpaint.channel(3);
for (int y=0; y<in.h; y++){
    for (int x=0; x<in.w; x++){
        ptd3[x] = ptr3[x];
    }
    ptd3 += in.w;
    ptr3 += in.w;
}
//处理4通道
float* ptr4 = mask.channel(0);
float* ptd4 = inpaint.channel(4);
for (int y=0; y<in.h; y++){
    for (int x=0; x<in.w; x++){
        ptd4[x] = ptr4[x];
    }
    ptd4 += in.w;
    ptr4 += in.w;
}

```



# 3 推断

```c++
ncnn::Extractor ex = mynet.create_extractor();  //创建extractor 多次创建互相之间独立不影响
//给ex输入mat
ex.input("data", in); 
ex.input("mask", mask);
ex.input("inpaint", inpaint);

//用mat接着输出
ncnn::Mat out;
ex.extract("add", out);
```



# 4 输出mat以便debug

```c++
void pretty_print(const ncnn::Mat& m, bool to255)
{
    FILE *fp = NULL;
    fp = fopen("data.txt", "w+"); // 刷新输入一个文本文件

    for (int q=0; q<1; q++){
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++){
            for (int x=0; x<m.w; x++){
                if(to255){   // 是否需要反归一化为0-255
                    fprintf(fp,"%d ", (int)(-(ptr[x]-1.0f)*127.5f));
                }else{
                    fprintf(fp,"%8.6f,", ptr[x]);
                }                
            }
            ptr += m.w;
            fprintf(fp,"\n");
        }
        fprintf(fp,"------------------------\n");
    }
    fclose(fp);
}

// 调用
pretty_print(out.channel(0),false);
```



# 5 保存mat为图片

```c++
void ncnn_image_save(ncnn::Mat& ncnn_img)
{
    // 目的是防止大于3通道以上的mat无法保存为图片 强制减少到3通道 并存入temp
    ncnn::Mat temp(ncnn_img.h, ncnn_img.w, 3, 4u);
    for (int q=0; q<3; q++){
        float* ptr = ncnn_img.channel(q);
        float* ptd = temp.channel(q);
        for (int y=0; y<ncnn_img.h; y++){
            for (int x=0; x<ncnn_img.w; x++){
                ptd[x] = ptr[x];
            }
            ptd += ncnn_img.w;
            ptr += ncnn_img.w;
        }
    }

    // 反归一化temp
    const float mean_vals[3] = {-1.0f, -1.0f, -1.0f};
    const float normalize_vals[3] = {127.5f, 127.5f, 127.5f};
    temp.substract_mean_normalize(mean_vals, normalize_vals); 

    // 转为ncnn的mat a并保存图片
    cv::Mat a(ncnn_img.h, ncnn_img.w, CV_8UC3);
    temp.to_pixels(a.data, ncnn::Mat::PIXEL_RGB);
    cv::imwrite("/home/gx/myproj/tensorflow2ncnn/build/examples/output2.png", a);
}
```

# 6 ncnn网络结构

按param的排序逐层构建layer，并通过layer重载的load_model方法顺序读取bin文件获得权重或数据

例如：memorydata读取固定数值，Convolution读取filter权重和bias

## 处理流程：

A 读入tensorflowLite的 json文件

B全网络    ->    分支网络结构           ->    卷积unit    ->    Convolution split elu relu interp等层csv写入器    ->   层权重bin写入器 

​                 ->    其他层 csv写入器

C csv转param（统计层数和blob个数 添加magic number等）



###  6.1 层权重bin写入器 

首先调整权重张量维度排布，tflite是 o-h-w-i  通过transpose(0,3,1,2）变为ncnn的格式 o-i-h-w

注意这里tflite的json文件中是直接将权重的一个float的写成了四个byte的0-255的数值，所以还需要加一个维度 

也即是先reshape将四个byte做成维度4 ：    temp = np.array(a).reshape(shape+(4,))

然后进行转置，维度4不动 ： transpose(0,3,1,2,4)

保存时因为tflite的json中是byte对应的整数0-255，故可以直接保存 用struct.pack('B',i)即可，都是小端数据不需要调整顺序

```python
# FROM o-h-w-i(tflite) TO o-i-h-w(ncnn)
def fix_weight_order(a,shape):
    temp = np.array(a).reshape(shape+(4,))
    print(temp.transpose(0,3,1,2,4).shape)
    return temp.transpose(0,3,1,2,4).flatten().tolist()

#save weight & bias data from tflite-json file to ncnn-bin file
def save_data_to_bin(bufferIndex, shape=(), savefilter=True, append=True):
    bufferData = model['buffers'][bufferIndex]['data']
    flag = 'ab' if append else 'wb'
    with open(netName+'.bin',flag) as f:
        if savefilter:
            i = 0x00000000   # flag-structure: weight data need to save as float type
            f.write(struct.pack('<I',i))
            bufferData = fix_weight_order(bufferData,shape)
        for i in bufferData: # sava bufferdata, no need flag-structure
            f.write(struct.pack('B',i))
    return
```



### 6.2 层csv写入器 

```python
def save_conv(name, indexTfLite, inputBlob, outputBlob):
    # 从json文件中获取卷积各项参数 
    op = model['subgraphs'][0]['operators']
    tensor = model['subgraphs'][0]['tensors']
    
    filterTensor = op[indexTfLite]['inputs'][1]
    biasTensor = op[indexTfLite]['inputs'][2]

    if 'dilation_w_factor' in op[indexTfLite]['builtin_options'].keys():
        dilation_w_factor = op[indexTfLite]['builtin_options']['dilation_w_factor']
    else:
        dilation_w_factor = '1'
        
    if 'stride_w' in op[indexTfLite]['builtin_options'].keys():
        stride_w = op[indexTfLite]['builtin_options']['stride_w']
    else:
        stride_w = '1'
	
    fliterShape = tuple(tensor[filterTensor]['shape'])
    
    param = ['0=' + str(fliterShape[0]),                    # num_output
             '1=' + str(fliterShape[1]),                    # kernel_w   （kernel_h default）
             '2=' + str(dilation_w_factor),                 # rate
             '3=' + str(stride_w),                          # stride
             '4=-233',                                      # same padding
             '5=1',                                         # has bias 需要从bin中读取bias
             '6=' + str(reduce(lambda x,y:x*y,fliterShape)) # filter weight 总长度 
            ]
    
    append_row_to_csv(['Convolution']+[name]+['1','1']+[inputBlob,name]+param)
    save_data_to_bin(tensor[filterTensor]['buffer'],fliterShape,savefilter=True)  # save filter's buffer (weight)
    save_data_to_bin(tensor[biasTensor]['buffer'],savefilter=False)   # save bias's   buffer 
    return outputBlob

def save_slice(name, inputBlob, outputBlob1, outputBlob2):
    # tensorflow中 切分张量的算子是split 而在ncnn中 split是复制blob 每个blob必须只有一个输出 所以二分支一定要用split分路
    # 注意ncnn切分参数 -23300代表0位是一个数组（-23301代表1位为数组） 数组的长度2 元素为-233代表均分
    append_row_to_csv(['Slice']+[name]+['1','2']+[inputBlob,outputBlob1,outputBlob2]+['-23300=2,-233,-233','1=0'])
    return outputBlob1, outputBlob2

def save_interp(name, inputBlob, outputBlob):                         # ResizeNearestNeighbor 2*
    # 注意插值的参数输入要求为一个float 必须写为 1=2.0 如果写成1=2会出错
    append_row_to_csv(['Interp']+[name]+['1','1']+[inputBlob,outputBlob]+['0=1','1=2.0','2=2.0']) 
    return outputBlob

def save_elu(name, inputBlob, outputBlob):
    # 注意ELU的大小写 
    append_row_to_csv(['ELU']+[name]+['1','1']+[inputBlob,outputBlob]+['0=1.0'])
    return outputBlob
    
def save_relu(name, inputBlob, outputBlob):
    # 注意ReLU大小写 
    append_row_to_csv(['ReLU']+[name]+['1','1']+[inputBlob,outputBlob])
    return outputBlob
    
def save_tanh(name, inputBlob, outputBlob):
    append_row_to_csv(['TanH']+[name]+['1','1']+[inputBlob,outputBlob])
    return outputBlob
    
def save_sigmoid(name, inputBlob, outputBlob):
    append_row_to_csv(['Sigmoid']+[name]+['1','1']+[inputBlob,outputBlob])
    return outputBlob

def save_mul(name, inputBlob1, inputBlob2, outputBlob):
    append_row_to_csv(['BinaryOp']+[name]+['2','1']+[inputBlob1,inputBlob2,outputBlob]+['0=2'])
    return outputBlob

def save_add(name, inputBlob1, inputBlob2, outputBlob):
    append_row_to_csv(['BinaryOp']+[name]+['2','1']+[inputBlob1,inputBlob2,outputBlob]+['0=0'])
    return outputBlob

def save_sub(name, inputBlob1, inputBlob2, outputBlob):
    append_row_to_csv(['BinaryOp']+[name]+['2','1']+[inputBlob1,inputBlob2,outputBlob]+['0=1'])
    return outputBlob

def save_split(name, inputBlob, outputBlob1, outputBlob2):
    append_row_to_csv(['Split']+[name]+['1','2']+[inputBlob,outputBlob1,outputBlob2])
    return outputBlob1, outputBlob2

def save_concat(name, inputBlob1, inputBlob2, outputBlob):
    # 注意concat的维度连接参数 0表示第三维 1表示h维 2表示w维
    append_row_to_csv(['Concat']+[name]+['2','1']+[inputBlob1,inputBlob2,outputBlob]+['0=0']) # 0 dim concat 1 h 2 w
    return outputBlob

def save_input(name, d0, d1, d2):
    append_row_to_csv(['Input']+[name]+['0','1']+[name]+['0='+str(d0)]+['1='+str(d1)]+['2='+str(d2)])
    return name

def save_memorydata(name, d0, d1, d2):
    append_row_to_csv(['MemoryData']+[name]+['0','1']+[name]+['0='+str(d0)]+['1='+str(d1)]+['2='+str(d2)])
    return name
```



### 6.3 卷积unit 分类型处理

```python
def save_conv_unit(unitType, indexTfLite, inputBlob, deconv, stage=''):
    # 添加命名规则
    def N(name):
        return stage+'/'+name+'_'+str(indexTfLite)
    
    x = inputBlob
    # deconv前面要加上interp插值
    if deconv:
        x = save_interp(('interp'), x, N('interp'))
        
    x = save_conv(N('conv2d'),indexTfLite, x, N('conv2d'))
    
    if unitType == 'TanH':
        x = save_tanh(N('tanh'),x,N('tanh'))
    else:  
        x,y = save_slice(N('slice'), x, N('conv2d')+'_A', N('conv2d')+'_B')

        if unitType == 'ELU':
            x = save_elu(N('elu'),x,N('elu'))
        if unitType == 'ReLU':
            x = save_relu(N('relu'),x,N('relu'))

        y = save_sigmoid(N('sigmoid'),y,N('sigmoid'))
        x = save_mul(N('mul'),x,y,N('mul'))   

    return x
```

### 6.4 分支网络结构

以构建一个网络分支为例

不使用钩子的情况下，需要预先知道tflite的模型里存储conv2d算子的编号（目的是按编号读取权重和bias）

```python
stage1 = [13,20,25,30,35, 40,45,50,55,60, 65,70,76,81,87, 92,97]
index = 1
x = 'inpaint'   # 本分支输入（上级的blob 这里都用字符串来表示blob）
for i in stage1:
    unitType = 'TanH' if index == 17 else 'ELU'       # 处理不同触发类型的卷积单元
    deconv = True if index in [13,15] else False      # 处理反卷积单元
    x = save_conv_unit(unitType,i,x,deconv,'stage1')  # 命名前缀加上stage1
    index += 1                                        # 用index计数来识别不同的卷积单元 防止tflite结构改变算子编号改变
```

特别关注反卷积问题

在tensorflow中，反卷积有两类实现方式：

1.使用转置卷积 

tf.nn.conv2d_transpose 参数表为：输入张量，卷积核（h,w,o,i 注意这里不是正卷积时候的H W I O），目标张量shape（也即是正卷积前的维度），striders（注意是正卷积时的步进），padding

转置卷积在数学逻辑上全等于 resize + conv2d。先进行的以strider计算的扩充性 resize， 特别注意这里resize是用了直接补零，而非最邻近或者线性插值。

转置卷积也可选进行参数训练

2.使用resize和conv2d算子组合

以本模型为例，主干网络有几个deconv（这特么其实也是反卷积，意义一样），构建模型的python代码中就直接用了resize_nearest_neighbor+conv2d。虽然这里用了最邻近插值，但鉴于反卷积不是一个数学上的严格概念，也即是不同的反卷积补零方式可以得到不同的结果，并没有唯一的解。 



### 6.5 csv转param

```python
def trans_csv_to_param():
    with open(netName+'.param','w') as p: 
        with open(netName+'.csv','r') as f:
            f_csv = csv.reader(f)

            blobs = []
            outputcontent = ''
            row_index = 0
            for row in f_csv:   ## read rows in csv
                # normal lines <layertype   layername   inputNum outputNum inputBlob outputBlob params>
                col_index = 0
                for col in row:
                    if col_index == 0:                # layertype
                        outputcontent += '%-16s ' % (col)
                    elif col_index == 1:              # layername
                        outputcontent += '%-32s ' % (col)
                    elif col_index == 2:              # inputNum
                        inputBlobNum = int(col)
                        outputcontent += ' %s' % (col)
                    elif col_index == 3:              # outputNum
                        outputBlobNum = int(col)
                        outputcontent += ' %s' % (col)
                    elif col_index > 3 and col_index <= 3+inputBlobNum+outputBlobNum:  # inputBlob outputBlob
                        blobs.append(col)
                        outputcontent += ' %s' % (col)
                    else:                             # params
                        outputcontent += ' %s' % (col)
                    col_index += 1
                outputcontent += '\n'
                row_index += 1  ## end of read rows in csv
            outputcontent = '7767517\n'+'%s '%(row_index)+'%s\n'%(len(list(set(blobs))))+outputcontent
            p.write(outputcontent)
```

# 7 ncnn不支持的自定义层实现原理

## 基础张量运算层

### transpose

考虑到可读性，实现转置需要单独对不同维度数的张量编写（否则要上递归）

以四维卷积为例，原张量shape为 （1，2，3，4）

```python
a = np.arange(24).reshape(1,2,3,4)
[[[[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]]

  [[12 13 14 15]
   [16 17 18 19]
   [20 21 22 23]]]]
```

转置目标维度表为：（2，1，0，3）也即是将第一维换到第三维，此处是维度编号和shape注意区分

实现转置函数如下：arr为原张量，dimtable为转置目标维度表

```python
def transpose(arr,dimtable):
    farr = arr.flatten() # 先将原张量摊平 赋值给farr 方便以偏移量直接索引 在c代码中可以直接用指针实现无需此过程
    
    shape = arr.shape    # ori 原始shape 1，2，3，4
    
    r_shape = []         # now 目标shape 3，2，1，4
    for i in dimtable:
        r_shape.append(shape[i])
        
    res = []             # 开辟新shape的张量 按顺序append 等同于c代码指针按顺序处理
    shift = [shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1]  # 原shape偏移表 => [24,12,4,1]
    
    for i in range(r_shape[0]):                  # 新shape第一维度
        for j in range(r_shape[1]):              # 新shape第二维度
            for m in range(r_shape[2]):          # 新shape第三维度
                for n in range(r_shape[3]):      # 新shape第四维度
                    step = 0
                    step += i*shift[dimtable[0]] # 读目标维度表 查原shape偏移表 shift[2] = 4
                    step += j*shift[dimtable[1]] # 读目标维度表 查原shape偏移表 shift[1] = 12                   
                    step += m*shift[dimtable[2]] # 读目标维度表 查原shape偏移表 shift[0] = 24    
                    step += n*shift[dimtable[3]] # 读目标维度表 查原shape偏移表 shift[3] = 1    
                    res.append(farr[step])       # 用step来偏移 从扁平张量farr索引 c代码用指针偏移直接实现
    return np.array(res).reshape(r_shape)
```

test 结果

```python
print( transpose(a,(2,1,0,3)) )

[[[[ 0  1  2  3]]
  [[12 13 14 15]]]
 [[[ 4  5  6  7]]
  [[16 17 18 19]]]
 [[[ 8  9 10 11]]
  [[20 21 22 23]]]]
```



### extract_image_patches

从图像张量中，抽取patch，并将抽取到的patch和通道一并摊平

例如：shape:(1,10,10,3)的一个10*10图像

```python
n = 10
images = [[[[x*n + y + 1, x*n + y + 1, x*n + y + 1] for y in range(n)] for x in range(n)]]
print(images)
images = np.array(images,  dtype=np.int32)

[[[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]], [[11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19], [20, 20, 20]], [[21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25], [26, 26, 26], [27, 27, 27], [28, 28, 28], [29, 29, 29], [30, 30, 30]], [[31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37], [38, 38, 38], [39, 39, 39], [40, 40, 40]], [[41, 41, 41], [42, 42, 42], [43, 43, 43], [44, 44, 44], [45, 45, 45], [46, 46, 46], [47, 47, 47], [48, 48, 48], [49, 49, 49], [50, 50, 50]], [[51, 51, 51], [52, 52, 52], [53, 53, 53], [54, 54, 54], [55, 55, 55], [56, 56, 56], [57, 57, 57], [58, 58, 58], [59, 59, 59], [60, 60, 60]], [[61, 61, 61], [62, 62, 62], [63, 63, 63], [64, 64, 64], [65, 65, 65], [66, 66, 66], [67, 67, 67], [68, 68, 68], [69, 69, 69], [70, 70, 70]], [[71, 71, 71], [72, 72, 72], [73, 73, 73], [74, 74, 74], [75, 75, 75], [76, 76, 76], [77, 77, 77], [78, 78, 78], [79, 79, 79], [80, 80, 80]], [[81, 81, 81], [82, 82, 82], [83, 83, 83], [84, 84, 84], [85, 85, 85], [86, 86, 86], [87, 87, 87], [88, 88, 88], [89, 89, 89], [90, 90, 90]], [[91, 91, 91], [92, 92, 92], [93, 93, 93], [94, 94, 94], [95, 95, 95], [96, 96, 96], [97, 97, 97], [98, 98, 98], [99, 99, 99], [100, 100, 100]]]]

```

python代码实现如下：

```python
def extract_image_patches(images,sizes,strides,rates,padding):
    farr = images.flatten()              # 先将原张量摊平 赋值给farr 方便以偏移量直接索引 在c代码中可以直接用指针实现无需此过程
    padding_row_num = (int) ((images.shape[1] + strides[1] - 1) / strides[1])        # 求pad矩阵的 行数 = 2
    padding_col_num = (int) ((images.shape[2] + strides[2] - 1) / strides[2])        # 求pad矩阵的 列数 = 2
    padding_cha_num = sizes[1]*sizes[2]*images.shape[3]                              # 求pad矩阵的 通道 = 4×4×3 = 48

    res = [] # 空列表 按顺序append 等同于c代码指针按顺序处理
    
    # 处理pad矩阵中每个pad
    for i in range(padding_row_num):
        for j in range(padding_col_num):
            padding_anchor = j*strides[2] + i*strides[1]*images.shape[2]      # 在原张量中找到pad的左上角所在的全局偏移量
            
            # 单独处理一个pad 按照sizes处理pad局部坐标 m,n
            for m in range(sizes[1]):
                for n in range(sizes[2]): 
                    # 在原张量内定位待处理局部点 和padding_anchor拼接为局部点的全局偏移量pros_pointer
                    pros_pointer = padding_anchor + n*rates[2] + m*rates[1]*images.shape[2]
                    pros_pointer *= images.shape[3]
					# 判断是否行列越界 这里默认SAME补零了 如需要VALID另行处理
                    isOutRow = (int)(padding_anchor / images.shape[2]) + m > images.shape[1]-1
                    isOutCol = padding_anchor % images.shape[2] + n > images.shape[2]-1
                    if isOutRow or isOutCol:
                        for ic in range(images.shape[3]): # 将每个通道摊平 append进res
                            res.append(0)
                        continue
                    # 处理非越界的正常状况 每个通道摊平 从farr中用偏移量寻址
                    for ic in range(images.shape[3]):
                        res.append(farr[pros_pointer+ic])
    return np.array(res).reshape(1,padding_row_num,padding_col_num,padding_cha_num)  # 返回时reshape c代码用指针 无所谓
```

运行得到一个4个patch 每个patch都是4×4×3 = 48 

```python
with tf.Session() as sess:
    res = tf.extract_image_patches(images=images,sizes=[1,4,4,1],strides=[1,7,7,1],rates=[1,1,1,1],padding='SAME').eval()
    print(res)
    print(res.shape)

[[[[  1   1   1   2   2   2   3   3   3   4   4   4  11  11  11  12  12
     12  13  13  13  14  14  14  21  21  21  22  22  22  23  23  23  24
     24  24  31  31  31  32  32  32  33  33  33  34  34  34]
   [  8   8   8   9   9   9  10  10  10   0   0   0  18  18  18  19  19
     19  20  20  20   0   0   0  28  28  28  29  29  29  30  30  30   0
      0   0  38  38  38  39  39  39  40  40  40   0   0   0]]

  [[ 71  71  71  72  72  72  73  73  73  74  74  74  81  81  81  82  82
     82  83  83  83  84  84  84  91  91  91  92  92  92  93  93  93  94
     94  94   0   0   0   0   0   0   0   0   0   0   0   0]
   [ 78  78  78  79  79  79  80  80  80   0   0   0  88  88  88  89  89
     89  90  90  90   0   0   0  98  98  98  99  99  99 100 100 100   0
      0   0   0   0   0   0   0   0   0   0   0   0   0   0]]]]
shape:(1, 2, 2, 48)
```



## 组合功能层

### extract_image_patches reshape transpose  （ERT）

输入： dim=3的ncnn::Mat   （隐含原始图片shape）

参数： 0 = sizes

​			1 = strides

输出： dim=1的ncnn::Mat   输出作为filter 因为是4维的ncnn不支持  将shape作为参数直接写到自定义的二输入卷积中



处理流程：

输入Mat  input（64，85，1）                  

------

@ERT (0=3,1=1)

-----

in： input

< extract_image_patches(input ,3,1,1,'SAME') >

out： res（dim=1） shape =（64，85，9）               inputElementN = 64×85×9

---

reshape直接计算 shape = （5440，3，3， inputElementN/3/3/5440）

---

in： res  shape = （5440，3，3，1）

< transpose >

out： res（dim=1）

---

输出Mat   res（5440）
























