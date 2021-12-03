from tqdm.notebook import tqdm
from train_util import train

def alpha_weight(step):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af
        
def pseudo_labeling(model, train_loader, test_loader, optimizer, loss_fn):
    EPOCHS = 3
    step = 100 
    model.train()
    for epoch in tqdm_notebook(range(EPOCHS)):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):
            model.eval()
            output_unlabeled = model(x_unlabeled)
            output_unlabeled = output_unlabeled.cpu().numpy()
            pseudo_label = output_unlabeled > 0.5
            pseudo_label = pseudo_label.astype(int)
            
            model.train()          
            output = model(x_unlabeled)
            unlabeled_loss = alpha_weight(step) * loss_fn(output, pseudo_labeled)   
            
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            
            if not batch_idx % 100:
                train(model, train_loader, optimizer, loss_fn, scheduler)
                step += 100