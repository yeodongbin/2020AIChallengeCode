import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from efficientnet_pytorch import EfficientNet

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import MaskRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from custom_model.faster_rcnn import fasterrcnn_resnet50_fpn
from custom_model.mask_rcnn import maskrcnn_resnet50_fpn

def get_model_instance_segmentation_custom0(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    
    print("fasterrcnn_resnet50_fpn  custom call - 41,755,286 (resnet50) / 28,730,006 (resnet18) / 28,730,006 resnet / 22,463,126 / 오잉..light resnet : 22,468,758/ 19,333,398 / custom resent (64 쭉..) 17,664,662")
    
    return model

def get_model_instance_segmentation0(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    
    print("fasterrcnn_resnet50_fpn  custom call - 41,755,286 / ")
    
    return model

def get_model_instance_segmentation(num_classes):
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    #backbone.out_channels = 1280

    #anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                               aspect_ratios=((0.5, 1.0, 2.0),))
    
    #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                            output_size=1,
    #                                            sampling_ratio=2)

    #model = FasterRCNN(backbone,
    #              num_classes=num_classes,
    #               rpn_anchor_generator=anchor_generator,
    #               box_roi_pool=roi_pooler)

    print("fasterrcnn_resnet50_fpn call - 41,401,661 / 41,532,886")
    # 분류를 위한 입력 특징 차원을 얻습니다
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    #hidden_layer = 1
    # and replace the mask predictor with a new one
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   hidden_layer,
    #                                                   num_classes)
    
    return model

def get_model_instance_segmentation_custom1(num_classes):
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    model = maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    #backbone.out_channels = 1280

    #anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                               aspect_ratios=((0.5, 1.0, 2.0),))
    
    #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                            output_size=1,
    #                                            sampling_ratio=2)

    #model = FasterRCNN(backbone,
    #              num_classes=num_classes,
    #               rpn_anchor_generator=anchor_generator,
    #               box_roi_pool=roi_pooler)

    print("maskrcnn_resnet50_fpn custom call1 - resnet : 24,743,507 mobilenet : 87,366,291  squeezenet : 33,161,683 densnet : 43,702,739, resnet basicblock 3*3 -> 1*1 : 20,549,203  / basic : 20,543,571 / basicblock con1 : 20,195,411 / 채널 : 강제로 128 지정시 13,033,555 / 128 all 변경 : 9,465,555 ")
    # 분류를 위한 입력 특징 차원을 얻습니다
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    hidden_layer = 128
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model

def get_model_instance_segmentation2(num_classes):

    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    backbone = torchvision.models.mobilenet_v2(pretrained=False).features
    #backbone.out_channels = 1
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=1,
                                                sampling_ratio=2)

    model = FasterRCNN(backbone,
                  num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    print("mobilenet_v2 call2 - out_channels :1280, 19,540,921")
    # 분류를 위한 입력 특징 차원을 얻습니다
    #in_features = backbone
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    #hidden_layer = 1
    # and replace the mask predictor with a new one
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   hidden_layer,
    #                                                   num_classes)
    
    return model

def get_model_instance_segmentation4(num_classes):

    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    backbone = torchvision.models.squeezenet1_1(pretrained=False).features
    #backbone.out_channels = 1
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                              output_size=14,
                                                              sampling_ratio=2)
    model = MaskRCNN(backbone,
                  num_classes=num_classes,
                   box_roi_pool =roi_pooler,
                   mask_roi_pool = mask_roi_pooler
                   )

    #print("squeezenet1_0 call2 - out_channels :1280, 18,052,473 / 72M")
    #print("squeezenet1_0 call2 - out_channels :516, 4,862,777 / 19.5M")
    #print("squeezenet1_1 call2 - out_channels :516, 4,849,849 4,862,777 / 19.5M")
    print("squeezenet1_1 call2 - out_channels :256, 2,757,369 / 11M (15,000,000 / 15,000,000)")
    print("squeezenet1_1 call2 - out_channels :512, 4,808,441 / 19.2M (15,000,000)")
    print("squeezenet1_1 call2 - out_channels :512, 33,192,463 33,161,683 / 172M (15,000,000)")
    

    #
    # 분류를 위한 입력 특징 차원을 얻습니다
    #in_features = backbone
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    #hidden_layer = 1
    # and replace the mask predictor with a new one
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   hidden_layer,
    #                                                   num_classes)
    
    return model

def get_model_instance_segmentation5(num_classes):

    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    backbone = torchvision.models.densenet161(pretrained=False).features
    #backbone.out_channels = 1
    backbone.out_channels = 256

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=1,
                                                sampling_ratio=2)

    model = FasterRCNN(backbone,
                  num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    print("densenet161 call2 - out_channels :256, 28,506,873 / 150M")

    # 분류를 위한 입력 특징 차원을 얻습니다
    #in_features = backbone
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    
    #hidden_layer = 1
    # and replace the mask predictor with a new one
    #model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                   hidden_layer,
    #                                                   num_classes)
    
    return model


def get_model_instance_segmentation6(num_classes):

    backbone = torchvision.models.squeezenet1_1(pretrained=False).features
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=1,
                                                sampling_ratio=2)

    model = FasterRCNN(backbone,
                  num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    print("get_model_instance_segmentation6 call6 - out_channels :512, 4,808,441 / (15,000,000) ")
    
    return model
