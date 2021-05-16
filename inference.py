from tqdm import tqdm
from train import *
dm.setup()
# model = MeiYun.load_from_checkpoint("/home/yz/.kaggle/ckp/MeiYun-val_loss=0.7820.ckpt")
model.cuda()
preds, labels, labss = [], [], []
for imgs, labs in tqdm(dm.val_dataloader()):
    outputs = model.predict(imgs)
    preds += outputs
    labels += labs.tolist()
    labss.append(labs)
print(len(preds))
preds_decoded = [dm.decode(pred) for pred in preds]
preds_inchis = ['InChI=1S/' + pred for pred in preds_decoded]
inchis = []
for labs in labss:
	labs_decoded = [dm.decode(lab) for lab in labs]
	inchis += ['InChI=1S/' + lab for lab in labs_decoded]
from Levenshtein import distance
metric = []
for pred, inchi in zip(preds_inchis, inchis):
    metric.append(distance(pred, inchi))
print(np.mean(metric))
dm.val.InChI = preds_inchis
dm.val['ld'] = metric
dm.val = dm.val.drop(['InChI_1', 'InChI_text'], axis = 1)
dm.val.to_csv('submission.csv', index=False)


# sample_submission = pd.read_csv(path / 'sample_submission.csv')
# limit = int(0.005*len(sample_submission))
# test_images = sample_submission.image_id[:limit]
# test_images = test_images.apply(lambda i: get_image_path(i, path, mode="test"))

# ds = Dataset(test_images, train=False, trans=A.Compose([A.Resize(128,128)]))

# dl = torch.utils.data.DataLoader(ds, batch_size=100*6, num_workers=4*6, pin_memory=True, shuffle=False)
# preds = []
# for batch in tqdm(dl):
#     outputs = model.predict(batch)
#     preds += outputs
# preds_decoded = [dm.decode(pred) for pred in preds]
# sample_submission.InChI[:limit] = ['InChI=1S/'+pred for pred in preds_decoded]
# print(sample_submission)
# sample_submission.to_csv('submission.csv', index=False)