import numpy as np 

def train(model, iterator, optimizer, loss_fn, scheduler):
    model.train()
    total_loss = 0
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        mask = (x != 0).float()
        outputs = model(x, mask)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"\tTrain loss {total_loss / len(iterator)}")

def evaluate(model, iterator):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        for x, y in tqdm(iterator):
            mask = (x != 0).float()
            outputs = model(x, mask)
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    pred = pred > 0.5
    pred = pred.astype(int)
    print(f'accuracy: {accuracy_score(pred, true)}')
    print(f'precision: {precision_score(pred, true)}')
    print(f'recall: {recall_score(pred, true)}')

def pred_test(model, iterator):
    model.eval()
    pred = []
    with torch.no_grad():
        for x in tqdm(iterator):
            mask = (x != 0).float()
            outputs = model(x, mask)
            pred += outputs.cpu().numpy().tolist()
    return np.array(pred)

