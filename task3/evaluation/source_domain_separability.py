import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from shared.pacs_protocol import SOURCE_DOMAINS


def source_domain_separability(backbone,source_val,device):
    backbone.eval()
    features= {}
    smallest= min(len(source_val[d].dataset) for d in SOURCE_DOMAINS)

    with torch.no_grad():
        for domain in SOURCE_DOMAINS:
            domain_features= []

            for images, _ in source_val[domain]:
                images= images.to(device)
                feats= backbone(images)

                domain_features.append(feats.cpu())

            domain_features= torch.cat(domain_features,dim=0)
            features[domain]= domain_features[:smallest]

    x= []
    y= []

    for i, domain in enumerate(SOURCE_DOMAINS):
        x.append(features[domain].numpy())
        y.append(np.full(smallest,i))

    x= np.concatenate(x,axis=0)
    y= np.concatenate(y,axis=0)

    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.30, random_state=6304, stratify=y)

    probe= LogisticRegression(C=1,max_iter=2000)

    probe.fit(x_train,y_train)

    predictions= probe.predict(x_test)

    return accuracy_score(y_test,predictions)