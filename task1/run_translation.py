import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from task1.data.stl10 import load_stl10,load_split,IMAGENET_NORMALIZE,CLIP_NORMALIZE
from task1.models.backbones import load_backbones,build_clip_text_features
from task1.models.linear_heads import load_linear_head
from task1.transforms import TranslationDataset
from task1.evaluation.metrics import accuracy,prediction_consistency

DIRECTIONS= {
    "right":lambda d:(d,0),
    "left":lambda d:(-d,0),
    "down":lambda d:(0,d),
    "up":lambda d:(0,-d)
}

DISPLACEMENTS= [0,8,16,32]


def predict_linear(backbone,head,loader,device,is_clip=False):
    predictions= []
    labels_all= []

    backbone.eval()
    head.eval()

    with torch.no_grad():
        for images,labels in loader:
            images= images.to(device)

            if is_clip:
                features= backbone.encode_image(images)
                features= features / features.norm(dim=-1,keepdim=True)
            else:
                features= backbone(images)

            logits= head(features.float())
            predictions.append(logits.argmax(dim=1).cpu())
            labels_all.append(labels.cpu())

    return torch.cat(predictions),torch.cat(labels_all)


def predict_clip_zeroshot(clip_model,text_features,loader,device):
    predictions= []
    labels_all= []

    clip_model.eval()

    with torch.no_grad():
        for images,labels in loader:
            images= images.to(device)
            image_features= clip_model.encode_image(images)
            image_features= image_features / image_features.norm(dim=-1,keepdim=True)
            logits= clip_model.logit_scale.exp() * image_features @ text_features.T
            predictions.append(logits.argmax(dim=1).cpu())
            labels_all.append(labels.cpu())

    return torch.cat(predictions),torch.cat(labels_all)


def make_loader(dataset,indices,dx,dy,normalize,batch_size,num_workers):
    translated= TranslationDataset(dataset,indices,dx,dy,normalize)
    return DataLoader(translated,batch_size=batch_size,shuffle=False,num_workers=num_workers)


def save_results(results,output_path):
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    with open(output_path,"w") as f:
        json.dump(results,f,indent=2)


def main(data_root,split_path,checkpoint_dir,output_path,batch_size=64,num_workers=2):
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    _,test_dataset= load_stl10(data_root,download=False)
    _,_,test_indices= load_split(split_path)

    resnet,vit,clip_model= load_backbones(device)

    resnet_head= load_linear_head("resnet",os.path.join(checkpoint_dir,"resnet_linear_head.pt"),device)
    vit_head= load_linear_head("vit",os.path.join(checkpoint_dir,"vit_linear_head.pt"),device)
    clip_head= load_linear_head("clip",os.path.join(checkpoint_dir,"clip_linear_head.pt"),device)
    text_features= build_clip_text_features(clip_model,test_dataset.classes,device)

    models= {
        "ResNet-50":(resnet,resnet_head,IMAGENET_NORMALIZE,False,False),
        "ViT-B-16":(vit,vit_head,IMAGENET_NORMALIZE,False,False),
        "CLIP Linear":(clip_model,clip_head,CLIP_NORMALIZE,True,False),
        "CLIP Zero-shot":(clip_model,None,CLIP_NORMALIZE,True,True)
    }

    if os.path.exists(output_path):
        with open(output_path,"r") as f:
            results= json.load(f)
        print("Loaded existing results. Completed entries will be skipped.")
    else:
        results= {}

    for model_name,(backbone,head,normalize,is_clip,is_zeroshot) in models.items():
        results.setdefault(model_name,{})

        clean_loader= make_loader(test_dataset,test_indices,0,0,normalize,batch_size,num_workers)

        if is_zeroshot:
            clean_predictions,labels= predict_clip_zeroshot(clip_model,text_features,clean_loader,device)
        else:
            clean_predictions,labels= predict_linear(backbone,head,clean_loader,device,is_clip=is_clip)

        for displacement in DISPLACEMENTS:
            key= str(displacement)

            if key in results[model_name]:
                print(model_name,displacement,"already done; skipping")
                continue

            direction_accuracies= []
            direction_consistencies= []

            for direction,direction_fn in DIRECTIONS.items():
                dx,dy= direction_fn(displacement)
                loader= make_loader(test_dataset,test_indices,dx,dy,normalize,batch_size,num_workers)

                if is_zeroshot:
                    predictions,current_labels= predict_clip_zeroshot(clip_model,text_features,loader,device)
                else:
                    predictions,current_labels= predict_linear(backbone,head,loader,device,is_clip=is_clip)

                direction_accuracies.append(accuracy(predictions,current_labels))
                direction_consistencies.append(prediction_consistency(clean_predictions,predictions))

            results[model_name][key]= {
                "accuracy":float(sum(direction_accuracies) / len(direction_accuracies)),
                "consistency":float(sum(direction_consistencies) / len(direction_consistencies)),
                "direction_accuracy":{name:float(value) for name,value in zip(DIRECTIONS.keys(),direction_accuracies)},
                "direction_consistency":{name:float(value) for name,value in zip(DIRECTIONS.keys(),direction_consistencies)}
            }

            save_results(results,output_path)
            print(model_name,displacement,results[model_name][key])

    print("Saved:",output_path)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--data_root",required=True)
    parser.add_argument("--split_path",required=True)
    parser.add_argument("--checkpoint_dir",required=True)
    parser.add_argument("--output_path",required=True)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--num_workers",type=int,default=2)
    args= parser.parse_args()

    main(
        data_root=args.data_root,
        split_path=args.split_path,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
