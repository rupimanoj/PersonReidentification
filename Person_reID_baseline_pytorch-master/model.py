import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
    
class ft_attr_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_attr_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.labels = ['age' ,'backpack' ,'bag', 'handbag', 'downcolor', 'upcolor' ,'clothes', 'down',
                                                     'up', 'hair' ,'hat', 'gender']            
        for attr in self.labels:
            name = 'classifier'+str(attr)
            if(attr == 'age'):
                setattr(self, name, ClassBlock(2048, 4))
            elif(attr == 'upcolor'):
                setattr(self, name, ClassBlock(2048, 8))
            elif(attr == 'downcolor'):
                setattr(self, name, ClassBlock(2048, 9))
            else:
                setattr(self, name, ClassBlock(2048, 2))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        y = []
        y.append(self.classifier(x))
        for attr in self.labels:
            name = 'classifier'+str(attr)
            c = getattr(self,name)
            predict_attr = c(x)
            y.append(predict_attr)
        y.append(x)
        return y

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_attr_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num)
        self.labels = ['age' ,'backpack' ,'bag', 'handbag', 'downcolor', 'upcolor' ,'clothes', 'down',
                                                     'up', 'hair' ,'hat', 'gender']            
        for attr in self.labels:
            name = 'classifier'+str(attr)
            if(attr == 'age'):
                setattr(self, name, ClassBlock(1024, 4))
            elif(attr == 'upcolor'):
                setattr(self, name, ClassBlock(1024, 8))
            elif(attr == 'downcolor'):
                setattr(self, name, ClassBlock(1024, 9))
            else:
                setattr(self, name, ClassBlock(1024, 2))

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x1 = self.classifier(x)
        y = []
        y.append(x1)
        for attr in self.labels:
            name = 'classifier'+str(attr)
            c = getattr(self,name)
            predict_attr = c(x)
            y.append(predict_attr)
        y.append(x)
        return y
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #print("Tensor size here:", x.size())
        x = self.avgpool(x)
        x1 = x
        x1 = x1.squeeze()
        x1 = x1.view(x1.size(0), -1)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        y.append(x1)
        return y

    
# Part Model proposed in Yifan Sun etal. (2018)
class PCB_attr(nn.Module):
    def __init__(self, class_num ):
        super(PCB_attr, self).__init__()
        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        model_undis_ft = models.resnet50(pretrained=True)
        self.model_undis = model_undis_ft
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.layer4_original = self.model.layer4
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.labels = ['age' ,'backpack' ,'bag', 'handbag', 'downcolor', 'upcolor' ,'clothes', 'down',
                                                     'up', 'hair' ,'hat', 'gender']
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))
            
        for attr in self.labels:
            name = 'classifier'+str(attr)
            if(attr == 'age'):
                setattr(self, name, ClassBlock(2048, 4))
            elif(attr == 'upcolor'):
                setattr(self, name, ClassBlock(2048, 8))
            elif(attr == 'downcolor'):
                setattr(self, name, ClassBlock(2048, 9))
            else:
                setattr(self, name, ClassBlock(2048, 2))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x_attr = x
        x = self.model.layer4(x)
        #print("before layer4 size here:", x_attr.size())
        x_attr = self.layer4_original(x_attr)
        #print("after layer4 size here:", x_attr.size())
        x_attr = self.global_pool(x_attr)
        x_attr = x_attr.squeeze()
        #print("after global poolingTensor size here:", x_attr.size())
        x = self.avgpool(x)
        
        #part pooling embeddings for extractions
        x1 = x
        x1 = x1.squeeze()
        x1 = x1.view(x1.size(0), -1)
        #attributes embeddings for extraction
        x2 = x_attr
        x2 = x2.squeeze()
        x2 = x2.view(x2.size(0), -1)
        x3 = torch.cat((x1, x2), 1)
        #print(x1.size(), "   ", x2.size(),"  ", x3.size())
        x = self.dropout(x)
        part = {}
        predict = {}
        predict_attr = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
            
        y = []
        for i in range(self.part):
            y.append(predict[i])
        
        i = 0
        for attr in self.labels:
            #print("feed forward for attr ", attr,"   ")
            name = 'classifier'+str(attr)
            c = getattr(self,name)
            predict_attr[i] = c(x_attr)
            #print(predict_attr[i])
            i += 1
            #print("end of attribute feed forward")

        for i in range(len(self.labels)):
            y.append(predict_attr[i])

        y.append(x3)
        #print("forward feed ouput size:", len(y))
        return y
    
# Part Model proposed in Yifan Sun etal. (2018)
class PCB_attr_resnet(nn.Module):
    def __init__(self, class_num ):
        super(PCB_attr, self).__init__()
        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        model_undis_ft = models.resnet50(pretrained=True)
        self.model_undis = model_undis_ft
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.layer4_original = self.model_undis.layer4
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.labels = ['age' ,'backpack' ,'bag', 'handbag', 'downcolor', 'upcolor' ,'clothes', 'down',
                                                     'up', 'hair' ,'hat', 'gender']
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))
            
        for attr in self.labels:
            name = 'classifier'+str(attr)
            if(attr == 'age'):
                setattr(self, name, ClassBlock(2048, 4))
            elif(attr == 'upcolor'):
                setattr(self, name, ClassBlock(2048, 8))
            elif(attr == 'downcolor'):
                setattr(self, name, ClassBlock(2048, 9))
            else:
                setattr(self, name, ClassBlock(2048, 2))
        self.resnet_classifier = ClassBlock(2048, class_num)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x_attr = x
        x = self.model.layer4(x)
        #print("before layer4 size here:", x_attr.size())
        x_attr = self.layer4_original(x_attr)
        #print("after layer4 size here:", x_attr.size())
        x_attr = self.global_pool(x_attr)
        x_attr = x_attr.squeeze()
        #print("after global poolingTensor size here:", x_attr.size())
        x = self.avgpool(x)
        
        #part pooling embeddings for extractions
        x1 = x
        x1 = x1.squeeze()
        x1 = x1.view(x1.size(0), -1)
        #attributes embeddings for extraction
        x2 = x_attr
        x2 = x2.squeeze()
        x2 = x2.view(x2.size(0), -1)
        
        x = self.dropout(x)
        part = {}
        predict = {}
        predict_attr = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
            
        y = []
        for i in range(self.part):
            y.append(predict[i])
        
        i = 0
        for attr in self.labels:
            #print("feed forward for attr ", attr,"   ")
            name = 'classifier'+str(attr)
            c = getattr(self,name)
            predict_attr[i] = c(x_attr)
            #print(predict_attr[i])
            i += 1
            #print("end of attribute feed forward")

        for i in range(len(self.labels)):
            y.append(predict_attr[i])
        x_resnet = self.resnet_classifier(x_attr)
        y.append(x_resnet)
        y.append(x1)
        #print("forward feed ouput size:", len(y))
        return y
    
class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

# debug model structure
#net = ft_net(751)
net = ft_net_dense(751)
#print(net)
input = Variable(torch.FloatTensor(8, 3, 224, 224))
output = net(input)
print('net output size:')
print(output.shape)
