
import torch
from data_utils import *
from nn_image_to_text import NN
from tqdm import tqdm
import os
import random
import transformers

SEED = 1
LR = 1e-4
EPOCHS = 21
DEVICE = "cuda"

dataset = get_dataset()
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
img_transform = get_transform()

for lr in [1e-3]:
    for ball in [1]:
        for p_drop in [0.3]:

            model = NN(ball=ball, p_drop=p_drop).to(DEVICE)
            checkpoint = torch.load("checkpoints/lr0.001_pdrop0.3_ball1_checkpoint_2.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            optim = torch.optim.AdamW(model.parameters(), lr=lr)
            num_training_steps = EPOCHS 
            lr_scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optim,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_training_steps//10
            )



            model.train()
            pbar = tqdm(range(EPOCHS))
            loss_history = {"train_loss":[], "val_loss":[]}

            for epoch in pbar:
                total_train = 0
                total_train_loss = 0
                total_val = 0
                total_val_loss = 0
                model.train()
                for i in tqdm(dataset["train"]):
                    encoded_verb_noun = model.tokenizer(i["verb noun"], return_tensors='pt').to(DEVICE)
                    optim.zero_grad()
                    image_list = i["positive_image_list"] + i["negative_image_list"]
                    image_batch = torch.zeros((len(image_list), 3, 224, 224)).to(DEVICE)
                    for ind, image in enumerate(image_list):
                        img_tensor = img_transform(image).to(DEVICE)
                        image_batch[ind] = img_transform(image).to(DEVICE)
                    text_feats, image_feats = model(encoded_verb_noun, image_batch)
                    label = torch.cat(
                                        (torch.zeros((len(i["positive_image_list"]), 1)),
                                        torch.ones((len(i["negative_image_list"]), 1))),
                                        dim=0
                                        ).to(DEVICE)

                    loss = model.loss(image_feats, text_feats, label)
                    loss.backward()
                    optim.step()
                        
                    total_train += 1
                    total_train_loss += loss.detach()
                    
                    pbar.set_description(f"loss = {loss.detach().cpu().numpy()}")

                model.eval()
                with torch.no_grad():
                    for i in dataset["test"]:
                        image_list = i["positive_image_list"] + i["negative_image_list"]
                        encoded_verb_noun = model.tokenizer(i["verb noun"], return_tensors='pt').to(DEVICE)
                        image_batch = torch.zeros((len(image_list), 3, 224, 224)).to(DEVICE)
                        for ind, image in enumerate(image_list):
                            img_tensor = img_transform(image).to(DEVICE)
                            image_batch[ind] = img_transform(image).to(DEVICE)
                        text_feats, image_feats = model(encoded_verb_noun, image_batch)
                        label = torch.cat(
                                            (torch.zeros((len(i["positive_image_list"]), 1)),
                                            torch.ones((len(i["negative_image_list"]), 1))),
                                            dim=0
                                            ).to(DEVICE)
                        loss = model.loss(text_feats, image_feats, label)
                        total_val += 1
                        total_val_loss += loss
                        
                                
                lr_scheduler.step()
                tqdm.write(f"epoch {epoch} | avg_train_loss = {total_train_loss/total_train} | avg_val_loss = {total_val_loss/total_val}")
                loss_history["val_loss"].append(float(total_train_loss/total_train))
                loss_history["train_loss"].append(float(total_val_loss/total_val))

                if epoch % 5 == 0 and epoch != 0:
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optim.state_dict(),
                                'loss_history': loss_history,
                                }, f"checkpoints/lr{lr}_pdrop{p_drop}_ball{ball}_checkpoint_{epoch//5}.pth")

            torch.save(model.state_dict(), "model.pt")
            #------------------------------------------------------EVAL------------------------------------------------
            model.eval()
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            with torch.no_grad():
                top_1 = 0
                top_5 = 0
                for i in tqdm(dataset["test"]):
                    top_image = [None]*5
                    top_dist = [0]*5
                    encoded_verb_noun = model.tokenizer(i["verb noun"], return_tensors='pt').to(DEVICE)
                    image_list = i["positive_image_list"] + i["negative_image_list"]
                    for image in image_list:
                        img_tensor = img_transform(image).to(DEVICE)
                        text_feats, image_feats = model(encoded_verb_noun, img_tensor)
                        dist = model.dist(text_feats, image_feats)
                        
                        for j in range(5):
                            if dist > top_dist[j]:
                                top_dist[j] = dist
                                top_image[j] = image
                                break

                        if dist >= model.ball and image in i["positive_image_list"]:
                            TP += 1
                        elif dist < model.ball and image in i["positive_image_list"]:
                            FN += 1
                        elif dist >= model.ball and image in i["negative_image_list"]:
                            FP += 1
                        elif dist < model.ball and image in i["negative_image_list"]:
                            TN += 1

                    for rank, image in enumerate(top_image):
                        if image in i["positive_image_list"] and rank == 0:
                            top_1 += 1
                            top_5 += 1
                            break
                        elif image in i["positive_image_list"]:
                            top_5 += 1
                            break
                top_1 /= len(dataset["test"])
                top_5 /= len(dataset["test"])
            print(f"lr={lr}, ball={ball}, p_drop={p_drop}")
            print(f"TP = {TP}")
            print(f"TN = {TN}")
            print(f"FP = {FP}")
            print(f"FN = {FN}")
            print(f"accuracy = {(TP+TN)/(TP+TN+FP+FN)}")
            print(f"top_1 = {top_1}")
            print(f"top_5 = {top_5}")







    
        








