import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from scipy.spatial.distance import cdist
import DataProvider.utils as DataUtils
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True, void_pixels=None):
	"""Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """

	assert (output.size() == label.size())

	labels = torch.ge(label, 0.5).float()

	num_labels_pos = torch.sum(labels)
	num_labels_neg = torch.sum(1.0 - labels)
	num_total = num_labels_pos + num_labels_neg

	output_gt_zero = torch.ge(output, 0).float()
	loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
		1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

	loss_pos_pix = -torch.mul(labels, loss_val)
	loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

	if void_pixels is not None:
		w_void = torch.le(void_pixels, 0.5).float()
		loss_pos_pix = torch.mul(w_void, loss_pos_pix)
		loss_neg_pix = torch.mul(w_void, loss_neg_pix)
		num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

	loss_pos = torch.sum(loss_pos_pix)
	loss_neg = torch.sum(loss_neg_pix)

	final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

	if size_average:
		final_loss /= np.prod(label.size())
	elif batch_average:
		final_loss /= label.size()[0]

	return final_loss


def fp_edge_loss(gt_edges, edge_logits):
	"""
    Edge loss in the first point network

    gt_edges: [batch_size, grid_size, grid_size] of 0/1
    edge_logits: [batch_size, grid_size*grid_size]
    """
	edges_shape = gt_edges.size()
	gt_edges = gt_edges.view(edges_shape[0], -1)

	loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges)

	return torch.mean(loss)


def fp_vertex_loss(gt_verts, vertex_logits):
	"""
    Vertex loss in the first point network

    gt_verts: [batch_size, grid_size, grid_size] of 0/1
    vertex_logits: [batch_size, grid_size**2]
    """
	verts_shape = gt_verts.size()
	gt_verts = gt_verts.view(verts_shape[0], -1)
	loss = F.binary_cross_entropy_with_logits(vertex_logits, gt_verts)
	return torch.mean(loss)


def poly_matching_loss_orderedinput(pred, gt, loss_type="L2"):
	if loss_type == "L2":
		dis = pred - gt
		dis = (dis ** 2).sum(2).sqrt().sum(1)
	elif loss_type == "L1":
		dis = pred - gt
		dis = torch.abs(dis).sum(2).sum(1)
	return dis


def heatmap_loss(pred, gt, grid_size, landmark_num):
	loss = 0
	if isinstance(pred, list):
		for pred_i in pred:
			dis = pred_i - gt
			loss += (dis ** 2).sum(3).sum(2).sqrt().sum(1)
	else:
		dis = pred - gt
		loss += (dis ** 2).sum(3).sum(2).sqrt().sum(1)
	return loss


def roi_loss(roipooled):
	roipooled_pos, roipooled_neg = roipooled
	background_max = torch.max(roipooled_pos[..., 1:], dim=-1)[0]
	forground_min = roipooled_neg[..., 0]
	loss = F.relu(background_max - forground_min)
	return loss


# img: N*C*H*W, res: N*1*H*W
def compute_grad_mag(img):
	filter_x = torch.cuda.FloatTensor([[1, 0, -1],
									   [2, 0, -2],
									   [1, 0, -1]], device=img.device)
	filter_y = torch.cuda.FloatTensor([[1, 2, 1],
									   [0, 0, 0],
									   [-1, -2, -1]], device=img.device)
	filter_x = filter_x.view((1, 1, 3, 3))
	filter_y = filter_y.view((1, 1, 3, 3))
	avg_img = torch.mean(img, dim=1, keepdim=True)
	grad_x = F.conv2d(avg_img, filter_x, padding=1)
	grad_y = F.conv2d(avg_img, filter_y, padding=1)
	grad_mag = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
	min_grad, max_grad = torch.min(grad_mag), torch.max(grad_mag)
	grad_mag = (grad_mag - min_grad) / (max_grad - min_grad)
	return grad_mag


# Deprecated
# img: N*C*H*W, res: N*1*H*W
def compute_grad_mag_v0(img):
	[N, C, H, W] = img.shape
	filter_x = torch.cuda.FloatTensor([[1, 0, -1],
									   [2, 0, -2],
									   [1, 0, -1]], device=img.device)
	filter_y = torch.cuda.FloatTensor([[1, 2, 1],
									   [0, 0, 0],
									   [-1, -2, -1]], device=img.device)
	filter_x = filter_x.view((1, 1, 3, 3))
	filter_y = filter_y.view((1, 1, 3, 3))
	grad_x = torch.zeros_like(img)
	grad_y = torch.zeros_like(img)
	res = torch.zeros([N, 1, H, W], dtype=torch.float32, device=img.device)
	for n in range(0, N):
		tmp_grad = torch.zeros([1, 1, H, W], dtype=torch.float32, device=img.device)
		for c in range(0, C):
			tmp_img = img[n, c].view((1, 1, H, W))
			grad_x = F.conv2d(tmp_img, filter_x, padding=1)
			grad_y = F.conv2d(tmp_img, filter_y, padding=1)
			tmp_grad += torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
		res[n][0] = torch.squeeze(tmp_grad)
	return res


def bilinear_interpolate(img, coords):
	[h, w] = img.shape[1:3]
	X = coords[:, :, 0]
	Y = coords[:, :, 1]

	X0 = torch.floor(X)
	X1 = X0 + 1
	Y0 = torch.floor(Y)
	Y1 = Y0 + 1

	W00 = (X1 - X) * (Y1 - Y)
	W01 = (X1 - X) * (Y - Y0)
	W10 = (X - X0) * (Y1 - Y)
	W11 = (X - X0) * (Y - Y0)

	X0 = torch.clamp(X0, 0, w - 1).long()
	X1 = torch.clamp(X1, 0, w - 1).long()
	Y0 = torch.clamp(Y0, 0, h - 1).long()
	Y1 = torch.clamp(Y1, 0, h - 1).long()

	I00 = X0 + Y0 * w
	I01 = X0 + Y1 * w
	I10 = X1 + Y0 * w
	I11 = X1 + Y1 * w

	flat_img = img.view(img.shape[0], -1)
	P00 = torch.gather(flat_img, 1, I00)
	P01 = torch.gather(flat_img, 1, I01)
	P10 = torch.gather(flat_img, 1, I10)
	P11 = torch.gather(flat_img, 1, I11)

	return W00 * P00 + W01 * P01 + W10 * P10 + W11 * P11


def get_grad_loss(img, grad_mag, pred):
	[h, w] = img.shape[2:4]
	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pt_grads = bilinear_interpolate(grad_mag, real_pred)
	# grad_loss = -pt_grads.mean(1)
	grad_loss = -pt_grads.mean()
	return grad_loss


def get_perceptual_loss_v1(vgg, img, pred_control, sample_img, sample_ctr):
	pt_num = sample_ctr.shape[0]
	pred_feat = vgg(img)
	sample_feat = vgg(sample_img.unsqueeze(0))
	r = 5
	perceptual_loss = torch.zeros(img.shape[0], dtype=torch.float32, device=img.device)
	for b in range(0, img.shape[0]):
		feat_num = 0
		for p1, p2 in zip(pred_control[b], sample_ctr):
			for f in range(1, len(sample_feat)):
				[h, w] = sample_feat[f].shape[2:4]
				x1 = torch.clamp(torch.round(p1[0] * w).int(), 0, w - 1)
				y1 = torch.clamp(torch.round(p1[1] * h).int(), 0, h - 1)
				x2 = torch.clamp(torch.round(p2[0] * w).int(), 0, w - 1)
				y2 = torch.clamp(torch.round(p2[1] * h).int(), 0, h - 1)
				# print(x1,y1,x2,y2)
				# tmp_pred_ctr = pred_control[b,:,:] * torch.cuda.FloatTensor([w,h], device=img.device)
				# tmp_pred_ctr = torch.round(tmp_pred_ctr).int()
				# tmp_sample_ctr = sample_ctr[:,:] * torch.cuda.FloatTensor([w,h], device=img.device)
				# tmp_sample_ctr = torch.round(tmp_sample_ctr).int()
				tmp_pred_feat = pred_feat[f][b, :, y1, x1]
				tmp_sample_feat = sample_feat[f][0, :, y2, x2]
				feat_num += tmp_pred_feat.shape[0]
				perceptual_loss[b] += torch.abs(tmp_pred_feat - tmp_sample_feat).sum()
		perceptual_loss[b] /= feat_num
	return perceptual_loss


def bilinear_interpolate_v2(img, coords):
	[n, c, h, w] = img.shape
	X = coords[:, :, 0] * w
	Y = coords[:, :, 1] * h

	X0 = torch.floor(X)
	X1 = X0 + 1
	Y0 = torch.floor(Y)
	Y1 = Y0 + 1

	W00 = (X1 - X) * (Y1 - Y)
	W01 = (X1 - X) * (Y - Y0)
	W10 = (X - X0) * (Y1 - Y)
	W11 = (X - X0) * (Y - Y0)

	X0 = torch.clamp(X0, 0, w - 1).long()
	X1 = torch.clamp(X1, 0, w - 1).long()
	Y0 = torch.clamp(Y0, 0, h - 1).long()
	Y1 = torch.clamp(Y1, 0, h - 1).long()

	I00 = X0 + Y0 * w
	I01 = X0 + Y1 * w
	I10 = X1 + Y0 * w
	I11 = X1 + Y1 * w

	flat_img = img.view(n, c, -1)
	P00 = torch.gather(flat_img, 2, I00.unsqueeze(1).repeat(1, c, 1))
	P01 = torch.gather(flat_img, 2, I01.unsqueeze(1).repeat(1, c, 1))
	P10 = torch.gather(flat_img, 2, I10.unsqueeze(1).repeat(1, c, 1))
	P11 = torch.gather(flat_img, 2, I11.unsqueeze(1).repeat(1, c, 1))

	return W00.unsqueeze(-1).repeat(1, 1, c) * torch.transpose(P00, 2, 1) + W01.unsqueeze(-1).repeat(1, 1,
																									 c) * torch.transpose(
		P01, 2, 1) + \
		   W10.unsqueeze(-1).repeat(1, 1, c) * torch.transpose(P10, 2, 1) + W11.unsqueeze(-1).repeat(1, 1,
																									 c) * torch.transpose(
		P11, 2, 1)


def get_perceptual_loss_v2(vgg, img, pred_control, sample_img, sample_ctr):
	perceptual_loss = torch.zeros(img.shape[0], dtype=torch.float32, device=img.device)
	pt_num = sample_ctr.shape[1]
	pred_feat = vgg(img)
	sample_feat = vgg(sample_img)
	feat_num = 0
	for f in range(1, len(sample_feat)):
		pred_ctr_feat = bilinear_interpolate_v2(pred_feat[f], pred_control)
		sample_ctr_feat = bilinear_interpolate_v2(sample_feat[f], sample_ctr)
		feat_dist = pred_ctr_feat - sample_ctr_feat
		feat_num += feat_dist.shape[-1]
		perceptual_loss += torch.abs(feat_dist).sum(2).sum(1)
		# perceptual_loss += (feat_dist**2).sum(2).sum(1).sqrt()
	return perceptual_loss / (feat_num * pt_num)


# CUDA out of memory
def get_perceptual_loss_v3(sample_name, pred_name, sample_ctr, pred_ctr):
	data_dir = '/home/yuhang/workspace/cvpr/knuckle_exp/all_knuckles'
	sample_img_feat = torch.load(os.path.join(data_dir, sample_name, sample_name + '.vggfeat'),
								 map_location=torch.device('cpu')).to(sample_ctr.device)
	sample_ctr_feat = bilinear_interpolate_v2(sample_img_feat, sample_ctr.unsqueeze(0))
	perceptual_loss = torch.zeros(pred_ctr.shape[0], dtype=torch.float32, device=sample_ctr.device)
	for i in range(0, pred_ctr.shape[0]):
		pred_img_feat = torch.load(os.path.join(data_dir, pred_name[i], pred_name[i] + '.vggfeat'),
								   map_location=torch.device('cpu')).to(sample_ctr.device)
		pred_ctr_feat = bilinear_interpolate_v2(pred_img_feat, pred_ctr[i].unsqueeze(0))
		perceptual_loss[i] = torch.dist(sample_ctr_feat, pred_ctr_feat)
		# del pred_img_feat
		# torch.cuda.empty_cache() # doesn't work
	return perceptual_loss


def expand_ctr(ctr, w, h, window_size=5):
	all_ctrs = []
	r = window_size // 2
	for dx in range(-r, r + 1):
		for dy in range(-r, r + 1):
			tmp_ctr = ctr[:, :] + torch.cuda.FloatTensor([dx, dy], device=ctr.device)
			all_ctrs.append(tmp_ctr)
	return torch.cat(all_ctrs, dim=1)


def get_perceptual_loss_v4(sample_gt_ctr, sample_img_feat, pred_ctr, pred_img_feat):
	device = pred_img_feat.device
	[n, c, h, w] = pred_img_feat.shape

	sample_gt_ctr = sample_gt_ctr.unsqueeze(0)
	sample_gt_ctr = sample_gt_ctr[:, :] * torch.cuda.FloatTensor([w, h], device=device)
	sample_gt_ctr = expand_ctr(sample_gt_ctr, w, h, window_size=5)
	sample_ctr_feat = bilinear_interpolate(sample_img_feat, sample_gt_ctr)

	pred_ctr = pred_ctr[:, :] * torch.cuda.FloatTensor([w, h], device=device)
	pred_ctr = expand_ctr(pred_ctr, w, h, window_size=5)
	pred_ctr_feat = bilinear_interpolate(pred_img_feat, pred_ctr)
	perceptual_loss = ((pred_ctr_feat - sample_ctr_feat) ** 2).sum(1).sqrt()
	return perceptual_loss / pred_ctr.shape[1]


# Use pre-computed contour features, support multiple samples
def get_perceptual_loss_v5(img, pred_ctr_feat, sample_ctr_feats):
	[n, c, h, w] = img.shape
	perceptual_loss = torch.zeros(n, dtype=torch.float32, device=img.device)
	for i in range(0, n):
		perceptual_loss[i] = torch.mean(
			torch.min(torch.mean(torch.abs(pred_ctr_feat[i] - sample_ctr_feats), 2), dim=0)[0])
	return perceptual_loss.mean()


# L1 dist of two features
def get_perceptual_loss_v6(pred_ctr_feat, sample_ctr_feat):
	return torch.abs(pred_ctr_feat - sample_ctr_feat).mean()


# L2 dist of two features
def get_perceptual_loss_v7(pred_ctr_feat, sample_ctr_feat):
	# return ((pred_ctr_feat - sample_ctr_feat) ** 2).mean()
	return ((pred_ctr_feat - sample_ctr_feat) ** 2).sum(2).sqrt().mean()


def get_std_loss(img, pred):
	[h, w] = img.shape[2:4]
	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pred_x, pred_y = real_pred[:, :, 0], real_pred[:, :, 1]
	pred_xp = torch.cat((pred_x[:, -1].unsqueeze(1), pred_x[:, 0:-1]), 1)
	pred_yp = torch.cat((pred_y[:, -1].unsqueeze(1), pred_y[:, 0:-1]), 1)
	dx = pred_x - pred_xp
	dy = pred_y - pred_yp
	# inter_dist = abs(dx)+abs(dy)
	# inter_dist = torch.sqrt(dx*dx + dy*dy)
	inter_dist = torch.sqrt(dx * dx + dy * dy + 1e-10)
	# inter_dist[:, 0] += 1e-4
	# std_loss = inter_dist.std(dim=1)
	std_loss = torch.sqrt(
		torch.sum(torch.pow(inter_dist - torch.mean(inter_dist, dim=1).unsqueeze(dim=1), 2), dim=1) / inter_dist.shape[
			1] + 1e-10)

	# Doesn't Work
	# zero_inds = (std_loss == 0).nonzero().squeeze()
	# if zero_inds.nelement() != 0:
	#     print(zero_inds)
	#     print(zero_inds.nelement())
	#     for ind in zero_inds:
	#         inter_dist[ind][0] += 1e-4
	#     std_loss = inter_dist.std(dim=1)

	# Doesn't Work
	# std_loss = torch.sqrt(inter_dist.var(dim=1))
	# std_loss[std_loss < 0.01] = 0.01

	# Doesn't Work
	# margin = torch.cuda.FloatTensor([0.01], device=img.device)
	# margin = margin.expand_as((std_loss))
	# hinge_std_loss = torch.max(std_loss, margin)
	# if not torch.all(torch.eq(std_loss, hinge_std_loss)):
	#     print(hinge_std_loss)

	return std_loss


def get_length_loss(img, pred):
	[h, w] = img.shape[2:4]
	# real_pred = pred[:,:] * torch.cuda.FloatTensor([w,h], device=img.device)
	pred_x, pred_y = pred[:, :, 0], pred[:, :, 1]
	prev_pred_x = torch.cat((pred_x[:, -1].unsqueeze(1), pred_x[:, 0:-1]), 1)
	prev_pred_y = torch.cat((pred_y[:, -1].unsqueeze(1), pred_y[:, 0:-1]), 1)
	dx = pred_x - prev_pred_x
	dy = pred_y - prev_pred_y
	inter_dist = abs(dx) + abs(dy)
	# l1_length = torch.sum(abs(dx), dim=1) + torch.sum(abs(dy), dim=1)
	# l2_length = torch.sum(torch.sqrt(dx ** 2 + dy ** 2), dim=1)
	l2_length = torch.sum(dx ** 2 + dy ** 2, dim=1)
	return l2_length


def get_internal_loss_v0(pred):
	pred_x, pred_y = pred[:, :, 0], pred[:, :, 1]
	prev_pred_x = torch.cat((pred_x[:, -1].unsqueeze(1), pred_x[:, 0:-1]), 1)
	prev_pred_y = torch.cat((pred_y[:, -1].unsqueeze(1), pred_y[:, 0:-1]), 1)
	next_pred_x = torch.cat((pred_x[:, 1:], pred_x[:, 0].unsqueeze(1)), 1)
	next_pred_y = torch.cat((pred_y[:, 1:], pred_y[:, 0].unsqueeze(1)), 1)
	dx = pred_x - prev_pred_x
	dy = pred_y - prev_pred_y
	length_loss = torch.sum(dx ** 2 + dy ** 2, dim=1)

	ddx = next_pred_x - 2 * pred_x + prev_pred_x
	ddy = next_pred_y - 2 * pred_y + prev_pred_y
	smoothness_loss = torch.sum(ddx ** 2 + ddy ** 2, dim=1)
	return length_loss, smoothness_loss, length_loss + smoothness_loss
	# return 0.001*length_loss + 0.4*smoothness_loss


def get_internal_loss_v1(img, pred):
	[h, w] = img.shape[2:4]
	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pred_x, pred_y = real_pred[:, :, 0], real_pred[:, :, 1]
	pred_xp = torch.cat((pred_x[:, -1].unsqueeze(1), pred_x[:, 0:-1]), 1)
	pred_yp = torch.cat((pred_y[:, -1].unsqueeze(1), pred_y[:, 0:-1]), 1)
	dx = pred_x - pred_xp
	dy = pred_y - pred_yp
	# point_dists = torch.sqrt(dx*dx + dy*dy)
	point_dists = abs(dx) + abs(dy)
	length_loss = torch.mean(point_dists, dim=1)
	std_loss = point_dists.var(dim=1)
	return length_loss, std_loss


def grad_and_std_loss_v0(img, grad_mag, pred):
	[h, w] = img.shape[2:4]
	# print(pred)
	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)

	pt_grads = bilinear_interpolate(grad_mag, real_pred)

	pred_x, pred_y = real_pred[:, :, 0], real_pred[:, :, 1]
	pred_xp = torch.cat((pred_x[:, -1].unsqueeze(1), pred_x[:, 0:-1]), 1)
	pred_yp = torch.cat((pred_y[:, -1].unsqueeze(1), pred_y[:, 0:-1]), 1)
	dx = pred_x - pred_xp
	dy = pred_y - pred_yp
	inter_dist = torch.sqrt(dx * dx + dy * dy)
	loss = 1 - pt_grads * 10
	# loss = pt_dists/10 + (1 - pt_grads*10)
	return loss.mean(1) + inter_dist.std(dim=1)


def grad_and_std_loss_v1(img, grad_mag, pred):
	[h, w] = img.shape[2:4]
	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pt_grads = bilinear_interpolate(grad_mag, real_pred)
	grad_loss = -pt_grads.mean(1)

	pred_x, pred_y = real_pred[:, :, 0], real_pred[:, :, 1]
	pred_xp = torch.cat((pred_x[:, -1].unsqueeze(1), pred_x[:, 0:-1]), 1)
	pred_yp = torch.cat((pred_y[:, -1].unsqueeze(1), pred_y[:, 0:-1]), 1)
	dx = pred_x - pred_xp
	dy = pred_y - pred_yp
	inter_dist = torch.sqrt(dx * dx + dy * dy)
	std_loss = inter_dist.std(dim=1)

	return grad_loss, 10 * grad_loss + std_loss


def dist_and_grad_loss(img, grad_mag, pred, gt, thre=10):
	[h, w] = img.shape[2:4]
	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	real_gt = gt[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)

	thre = 10
	# thre = 10 * math.sqrt(2.0/(h**2+w**2))

	pt_grads = bilinear_interpolate(grad_mag, real_pred)

	# pt_dists = torch.abs(real_pred - real_gt).sum(2)
	pt_dists = ((real_pred - real_gt) ** 2).sum(2).sqrt()

	mask = (pt_dists < thre).type(torch.cuda.FloatTensor)
	# mask[:,:] = 0
	# mask[:,0:25] = 1
	# mask[:,175:200] = 1

	# loss = pt_dists + torch.mul(mask, -pt_grads)
	# loss = torch.mul(mask, pt_dists) + torch.mul(1-mask, 1-pt_grads)
	loss = torch.mul(mask, pt_dists) - pt_grads
	# loss = 1-pt_grads/100

	return loss.mean(1)


def get_dist_loss(img, pred, target):
	[h, w] = img.shape[2:4]
	pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	target = target[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	dist_loss = ((pred - target) ** 2).sum(2).sqrt()
	# return dist_loss
	return dist_loss.mean()


def get_gvf_loss(img, u, v, init, pred):
	gvf_dx = bilinear_interpolate(u, init)
	gvf_dy = bilinear_interpolate(v, init)
	gvf_offset = torch.stack((gvf_dx, gvf_dy), dim=2)
	pred_offset = pred - init

	gvf_offset = (gvf_offset - torch.min(gvf_offset)) / (torch.max(gvf_offset) - torch.min(gvf_offset))
	pred_offset = (pred_offset - torch.min(pred_offset)) / (torch.max(pred_offset) - torch.min(pred_offset))
	gvf_loss = torch.abs(pred_offset - gvf_offset).sum(2)

	return gvf_loss
	# return gvf_loss.mean(1)


def get_gvf_loss_v2(img, u, v, init, pred):
	gvf_dx = bilinear_interpolate(u, pred)
	gvf_dy = bilinear_interpolate(v, pred)
	gvf_loss = torch.abs(gvf_dx) + torch.abs(gvf_dy)
	return gvf_loss
	# return gvf_loss.mean(1)


def gvf_and_dist_loss(img, u, v, init, pred, target):
	# gvf_loss = get_gvf_loss(img, u, v, init, pred)
	gvf_loss = get_gvf_loss_v2(img, u, v, init, pred)
	dist_loss = get_dist_loss(img, pred, target)
	thre = 5
	mask = (dist_loss > thre).type(torch.cuda.FloatTensor)
	loss = torch.mul(mask, dist_loss) + 1000 * gvf_loss
	# loss = torch.mul(mask, dist_loss) + torch.mul(1-mask, 1000*gvf_loss)
	return loss


# Doesn't work
# From pred to target, results in sparse pred points with GT
def get_partial_sup_loss_v1(img, grad_mag, pred, partial_target, ori_img):
	b, c, h, w = img.shape
	aligned_target = np.full(pred.shape, -1.0)

	for i in range(0, b):
		tmp_pred = pred[i].data.cpu().numpy() * [w, h]
		tmp_target = partial_target[i].data.cpu().numpy() * [w, h]
		target_pred_dist = cdist(tmp_target, tmp_pred, 'euclidean')
		closest_pred_idx = np.argmin(target_pred_dist, axis=1)
		tmp_target /= [w, h]
		print(closest_pred_idx)
		print(np.unique(closest_pred_idx).shape)
		for target_idx, pred_idx in enumerate(closest_pred_idx):
			aligned_target[i][pred_idx] = tmp_target[target_idx]
		if i == 0:
			vis_partial_target(ori_img[i], tmp_pred, tmp_target, closest_pred_idx)

	mask = (aligned_target != (-1, -1)).all(axis=2)
	print(mask.sum())
	mask = torch.cuda.BoolTensor(mask, device=img.device)
	point_dist = get_dist_loss(img, pred, torch.cuda.FloatTensor(aligned_target, device=img.device))
	dist_loss = torch.masked_select(point_dist, mask).mean()

	real_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	point_grad = bilinear_interpolate(grad_mag, real_pred)
	grad_loss = torch.masked_select(-point_grad, ~mask).mean()

	std_loss = get_std_loss(img, pred).mean()
	return dist_loss + grad_loss + std_loss


# For multiple contour segments
def get_partial_dist_loss_v2(img, grad_mag, pred, partial_target, ori_img):
	b, c, h, w = img.shape
	ori_pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	ori_partial_target = partial_target[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	aligned_target = torch.cuda.FloatTensor(pred.shape, device=img.device).fill_(-1)
	for i in range(0, b):
		tmp_pred = ori_pred[i].data.cpu().numpy()
		for c in range(0, ori_partial_target[i].shape[0]):
			tmp_target = ori_partial_target[i][c].data.cpu().numpy()
			target_pred_dist = cdist(tmp_target, tmp_pred, 'euclidean')
			closest_pred_idx = np.argmin(target_pred_dist, axis=1)
			pred_start_idx, pred_end_idx = closest_pred_idx[0], closest_pred_idx[-1]
			pred_target_dist = target_pred_dist.transpose()
			closest_target_idx = np.argmin(pred_target_dist, axis=1)
			if pred_start_idx > pred_end_idx:
				covered_pred_idx = list(range(pred_start_idx, pred.shape[1])) + list(range(0, pred_end_idx + 1))
			else:
				covered_pred_idx = list(range(pred_start_idx, pred_end_idx + 1))
			for pred_idx in covered_pred_idx:
				aligned_target[i][pred_idx] = partial_target[i][c][closest_target_idx[pred_idx]]
			# vis_partial_target(ori_img[i], tmp_pred, tmp_target, closest_pred_idx)

	mask = (aligned_target != torch.cuda.FloatTensor([-1, -1], device=img.device)).all(axis=2)
	# mask = torch.cuda.BoolTensor(mask, device=img.device)
	point_dist = get_dist_loss(img, pred, aligned_target)
	dist_loss = torch.masked_select(point_dist, mask).mean()

	point_grad = bilinear_interpolate(grad_mag, ori_pred)
	grad_loss = torch.masked_select(-point_grad, ~mask).mean()

	std_loss = get_std_loss(img, pred).mean()
	return dist_loss + grad_loss + std_loss


# For one discontinue contour
# The most straightforward way, but cdist has bug in backward when the tensor is too big
def get_partial_dist_loss_v3(img, pred, partial_target):
	n, c, h, w = img.shape
	dist_loss = torch.zeros(n, dtype=torch.float32, device=img.device)
	for i in range(0, n):
		if partial_target[i].sum() == 0.0:
			continue
		tmp_pred = pred[i]
		tmp_part_gt = partial_target[i]
		tmp_part_gt = tmp_part_gt[torch.nonzero(tmp_part_gt.sum(axis=1))].squeeze()
		dist_loss[i] = torch.min(torch.cdist(tmp_part_gt, tmp_pred), dim=1)[0].sum()
	return dist_loss.mean()


# For one discontinue contour
# Constructing a pseudo gt contour. If a pred point does not have a partial gt point, use itself as gt to make loss 0
def get_partial_dist_loss_v4(img, pred, partial_target):
	n, c, h, w = img.shape
	pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	partial_target = partial_target[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	dist_loss = torch.zeros(n, dtype=torch.float32, device=img.device)
	for i in range(0, n):
		if partial_target[i].sum() == 0.0:
			continue
		tmp_pred = pred[i]
		tmp_part_gt = partial_target[i]
		tmp_part_gt = tmp_part_gt[torch.nonzero(tmp_part_gt.sum(axis=1))].squeeze()

		tmp_pseudo_gt = tmp_pred.clone().detach()
		pred_pt_idx = torch.min(torch.cdist(tmp_part_gt, tmp_pred), dim=1)[1]
		tmp_pseudo_gt[pred_pt_idx] = tmp_part_gt
		dist_loss[i] = ((tmp_pred - tmp_pseudo_gt) ** 2).sum()
		# torch.nonzero(((tmp_pred - tmp_pseudo_gt) ** 2).sum(1)).shape
	# print(dist_loss)
	return dist_loss.mean()


# partial_target uses [-1,-1] as delimiter
# There is problem in processing the start and end points when the pred ctr is twisted
def get_partial_dist_loss_v5(img, pred, partial_target, ori_img):
	n, c, h, w = img.shape
	pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pseudo_gt = pred.detach().clone()
	for i in range(0, n):
		tmp_pred = pred[i].data.cpu().numpy()
		gt_segs = get_partial_gt_list(partial_target[i])
		for c in range(0, len(gt_segs)):
			gt_seg = gt_segs[c].data.cpu().numpy()
			gt_to_pred_dist = cdist(gt_seg, tmp_pred, 'euclidean')
			closest_gt_idx = np.argmin(gt_to_pred_dist, axis=0)
			closest_pred_idx = np.argmin(gt_to_pred_dist, axis=1)
			pred_start_idx, pred_end_idx = closest_pred_idx[0], closest_pred_idx[-1]
			if pred_start_idx > pred_end_idx:  # In case the gt_seg across the start point
				covered_pred_idx = list(range(pred_start_idx, pred.shape[1])) + list(range(0, pred_end_idx + 1))
			else:
				covered_pred_idx = list(range(pred_start_idx, pred_end_idx + 1))
			for pred_idx in covered_pred_idx:  # Use the gt coord to replace the pred coord
				pseudo_gt[i][pred_idx] = gt_segs[c][closest_gt_idx[pred_idx]]

			if len(covered_pred_idx) > 300:
				print(len(covered_pred_idx))
				vis_partial_target(ori_img[i], tmp_pred, gt_seg, covered_pred_idx, closest_gt_idx)

	# mask = (pseudo_gt != pred).all(axis=2)
	# all_dist = ((pseudo_gt - pred) ** 2).sum(2)
	# if mask.sum() == 0:
	#     dist_loss = 0
	# else:
	#     dist_loss = torch.masked_select(all_dist, mask).mean()
	dist_loss = torch.abs(pseudo_gt - pred).sum(2).mean()
	# dist_loss = ((pseudo_gt - pred) ** 2).sum(2).sqrt().mean() # Cannot backpropogate when dist is 0
	return dist_loss


# Find the closest pred point for each gt point, does not work
def get_partial_dist_loss_v6(img, pred, partial_target):
	n, c, h, w = img.shape
	pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pseudo_gt = pred.detach().clone()
	for i in range(0, n):
		if (partial_target[i]!=-1.0).sum() == 0.0:
			continue
		tmp_gt_seg = partial_target[i][partial_target[i].sum(dim=1) != -2.0]
		selected_pred_idx = torch.min(torch.cdist(tmp_gt_seg, pred[i]), dim=1)[1]
		pseudo_gt[i][selected_pred_idx] = tmp_gt_seg
		# vis_img = np.zeros((h, w, 3), np.uint8)
		# ctr = np.round(pseudo_gt[i].data.cpu().numpy()).astype(int)
		# for p in ctr:
		# 	vis_img[p[1], p[0]] = [0, 255, 255]
		# cv2.imshow('1', vis_img)
		# cv2.waitKey()
	dist_loss = torch.abs(pseudo_gt - pred).sum(2).mean()
	return dist_loss


def get_partial_gt_list(coords):
    gt_segs = list()
    start = 0
    for i in range(1, coords.shape[0]):
        if coords[i-1].sum() != -2.0 and (coords[i].sum() == -2.0 or i == coords.shape[0]-1):
            gt_segs.append(coords[start:i])
            start = i+1
    return gt_segs

def get_partial_dist_loss_v7(img, pred, partial_target, ori_img):
	n, c, h, w = img.shape
	pt_num = pred.shape[1]
	pred = pred[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	pseudo_gt = pred.detach().clone()
	for i in range(0, n):
		gt_segs = get_partial_gt_list(partial_target[i])
		for c in range(0, len(gt_segs)):
			gt_to_pred_dist = torch.cdist(gt_segs[c], pred[i])
			closest_gt_idx = torch.min(gt_to_pred_dist, dim=0)[1]
			closest_pred_idx = torch.min(gt_to_pred_dist, dim=1)[1]

			end1, end2 = closest_pred_idx[0], closest_pred_idx[-1]
			if end1 > end2:
				tmp = end1
				end1 = end2
				end2 = tmp

			### For knuckle and lung
			gt_seg_pt_num = gt_segs[c].shape[0]
			pred_idx_list1 = list(range(end1, end2 + 1))
			pred_idx_list2 = list(range(end2, pred.shape[1])) + list(range(0, end1 + 1))
			if abs(len(pred_idx_list1)-gt_seg_pt_num) < abs(len(pred_idx_list2)-gt_seg_pt_num):
				pred_idx_list = pred_idx_list1
			else:
				pred_idx_list = pred_idx_list2

			### For knee
			# pred_idx_list = list(range(end1, end2 + 1))

			sampled_gt_seg = DataUtils.sample_arc_point(gt_segs[c].data.cpu().numpy(), len(pred_idx_list))
			pseudo_gt[i][pred_idx_list] = torch.from_numpy(sampled_gt_seg.astype(np.float32)).to(img.device)
			# if len(pred_idx_list) > 300:
			# 	print(len(pred_idx_list))

		# show two contours
		# vis_img = np.zeros((h, w, 3), np.uint8)
		# gt_ctr = np.round(pseudo_gt[i].data.cpu().numpy()).astype(int)
		# pred_ctr = np.round(pred[i].data.cpu().numpy()).astype(int)
		# for p in range(0, pt_num):
		# 	vis_img[gt_ctr[p][1], gt_ctr[p][0]] = [0, 255, 0]
		# 	vis_img[pred_ctr[p][1], pred_ctr[p][0]] = [0, 255, 255]
		# 	cv2.imshow('1', vis_img)
		# 	cv2.waitKey()

	dist_loss = torch.abs(pseudo_gt - pred).mean()
	return dist_loss


def vis_partial_target(ori_img, tmp_pred, tmp_target, covered_pred_idx, closest_gt_idx):
	c, h, w = ori_img.shape
	vis_img = np.transpose(ori_img.data.cpu().numpy(), [1, 2, 0]).astype(np.uint8)
	tmp_real_pred = np.round(tmp_pred).astype(np.int32)
	tmp_real_target = np.round(tmp_target).astype(np.int32)
	for p in tmp_real_pred:
		vis_img[p[1], p[0]] = (255, 0, 0)
	for p in tmp_real_target:
		vis_img[p[1], p[0]] = (0, 255, 0)
	tmp_vis_img = vis_img.copy()
	for i, pred_idx in enumerate(covered_pred_idx):
		target_idx = closest_gt_idx[pred_idx]
		tmp_vis_img[tmp_real_pred[pred_idx][1], tmp_real_pred[pred_idx][0]] = (0, 0, 255)
		tmp_vis_img[tmp_real_target[target_idx][1], tmp_real_target[target_idx][0]] = (0, 0, 255)
	cv2.imshow('1', tmp_vis_img)
	cv2.waitKey()


def AffineFit(from_pts, to_pts):
	"""Fit an affine transformation to given point sets.
      More precisely: solve (least squares fit) matrix 'A'and 't' from
      'p ~= A*q+t', given vectors 'p' and 'q'.
      Works with arbitrary dimensional vectors (2d, 3d, 4d...).

      Written by Jarno Elonen <elonen@iki.fi> in 2007.
      Placed in Public Domain.

      Based on paper "Fitting affine and orthogonal transformations
      between two sets of points, by Helmuth Späth (2003)."""

	q = from_pts
	p = to_pts
	if len(q) != len(p) or len(q) < 1:
		print("from_pts and to_pts must be of same size.")
		return

	dim = len(q[0])  # num of dimensions
	if len(q) < dim:
		print("Too few points => under-determined system.")
		return

	# Make an empty (dim) x (dim+1) matrix and fill it
	c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
	for j in range(dim):
		for k in range(dim + 1):
			for i in range(len(q)):
				qt = list(q[i]) + [1]
				c[k][j] += qt[k] * p[i][j]

	# Make an empty (dim+1) x (dim+1) matrix and fill it
	Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
	for qi in q:
		qt = list(qi) + [1]
		for i in range(dim + 1):
			for j in range(dim + 1):
				Q[i][j] += qt[i] * qt[j]

	# Ultra simple linear system solver. Replace this if you need speed.
	def gauss_jordan(m, eps=1.0 / (10 ** 10)):
		"""Puts given matrix (2D array) into the Reduced Row Echelon Form.
         Returns True if successful, False if 'm' is singular.
         NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
         Written by Jarno Elonen in April 2005, released into Public Domain"""
		(h, w) = (len(m), len(m[0]))
		for y in range(0, h):
			maxrow = y
			for y2 in range(y + 1, h):  # Find max pivot
				if abs(m[y2][y]) > abs(m[maxrow][y]):
					maxrow = y2
			(m[y], m[maxrow]) = (m[maxrow], m[y])
			if abs(m[y][y]) <= eps:  # Singular?
				return False
			for y2 in range(y + 1, h):  # Eliminate column y
				c = m[y2][y] / m[y][y]
				for x in range(y, w):
					m[y2][x] -= m[y][x] * c
		for y in range(h - 1, 0 - 1, -1):  # Backsubstitute
			c = m[y][y]
			for y2 in range(0, y):
				for x in range(w - 1, y - 1, -1):
					m[y2][x] -= m[y][x] * m[y2][y] / c
			m[y][y] /= c
			for x in range(h, w):  # Normalize row y
				m[y][x] /= c
		return True

	# Augement Q with c and solve Q * a' = c by Gauss-Jordan
	M = [Q[i] + c[i] for i in range(dim + 1)]
	if not gauss_jordan(M):
		print("Error: singular matrix. Points are probably coplanar.")
		return

	# Make a result object
	class Transformation:
		"""Result object that represents the transformation
           from affine fitter."""

		def To_Str(self):
			res = ""
			for j in range(dim):
				str = "x%d' = " % j
				for i in range(dim):
					str += "x%d * %f + " % (i, M[i][j + dim + 1])
				str += "%f" % M[dim][j + dim + 1]
				res += str + "\n"
			return res

		def Transform(self, pt):
			res = [0.0 for a in range(dim)]
			for j in range(dim):
				for i in range(dim):
					res[j] += pt[i] * M[i][j + dim + 1]
				res[j] += M[dim][j + dim + 1]
			return res

	# return Transformation()
	return np.array(M)[:, 3:]


def show_affine_ctr(img, ctr1, ctr2, ctr3):
	img = img.data.cpu().numpy().transpose(1, 2, 0)
	img = (255 * img).astype(np.uint8)
	ctr1 = np.round(ctr1).astype(int)
	ctr2 = np.round(ctr2).astype(int)
	ctr3 = np.round(ctr3).astype(int)
	for i in range(0, ctr1.shape[0]):
		img[ctr1[i][1], ctr1[i][0]] = [0, 0, 255]  # sample_ctr
		img[ctr2[i][1], ctr2[i][0]] = [0, 255, 0]  # pred_ctr
		img[ctr3[i][1], ctr3[i][0]] = [0, 255, 255]  # pseudo_gt_ctr
	cv2.imshow('1', img)
	cv2.waitKey()


def get_affine_shape_loss(img, pred_ctr, sample_ctr):
	[n, c, h, w] = img.shape
	pred_ctr = pred_ctr[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	sample_ctr = sample_ctr[:] * torch.cuda.FloatTensor([w, h], device=img.device)
	pseudo_gt_ctr = torch.zeros(pred_ctr.shape, dtype=torch.float32, device=img.device)
	for i in range(0, n):
		from_pt = sample_ctr.data.cpu().numpy()
		to_pt = pred_ctr[i].data.cpu().numpy()
		M = AffineFit(from_pt, to_pt)
		pad_from_pt = np.hstack([from_pt, np.ones((from_pt.shape[0], 1))])
		pseudo_gt_ctr[i] = torch.from_numpy(np.dot(pad_from_pt, M)).to(img.device)
		# for p in range(0, from_pt.shape[0]):
		# pseudo_gt_ctr[i][p] = torch.FloatTensor(trn.Transform(from_pt[p]))
		# show_affine_ctr(img[i], from_pt, to_pt, pseudo_gt_ctr[i].data.cpu().numpy())
	shape_loss = ((pred_ctr - pseudo_gt_ctr) ** 2).sum(2).sqrt()
	return shape_loss.mean(1)


def get_tps_shape_loss_v1(img, pred_ctr, sample_ctr):
	[n, c, h, w] = img.shape
	pred_ctr = pred_ctr[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	sample_ctr = sample_ctr[:, :] * torch.cuda.FloatTensor([w, h], device=img.device)
	shape_loss = torch.zeros(img.shape[0], dtype=torch.float32, device=img.device)
	for i in range(0, n):
		vp, vq = pred_ctr[i], sample_ctr[i]
		m = vp.shape[0]
		differences = vq.unsqueeze(1) - vq.unsqueeze(0)
		dist = torch.sqrt(torch.sum(differences * differences, -1))
		K = torch.pow(dist, 2) * torch.log(dist)
		for p in range(0, m):
			K[p, p] = 0
		P = torch.cat([torch.ones((m, 1), device=img.device), vq], dim=1)
		L = torch.cat([torch.cat([K, P], dim=1), torch.cat([P.t(), torch.zeros((3, 3), device=img.device)], dim=1)],
					  dim=0)
		# L = torch.inverse(L)[0:m,0:m]
		L = np.linalg.inv(L.data.cpu().numpy())[0:m, 0:m]
		L = torch.from_numpy(L).to(img.device)
		xp, yp = vp[:, 0], vp[:, 1]
		shape_loss[i] = (torch.mm(torch.mm(xp.unsqueeze(0), L), xp.unsqueeze(0).t()) + torch.mm(
			torch.mm(yp.unsqueeze(0), L), yp.unsqueeze(0).t())) / (8 * np.pi)
		# show_affine_ctr(img[i], sample_ctr[i].data.cpu().numpy(), pred_ctr[i].data.cpu().numpy(), pred_ctr[i].data.cpu().numpy())
	return shape_loss


def get_tps_shape_loss_v2(img, pred_ctr, sample_tps_inv_l):
	[n, c, h, w] = img.shape
	pt_num = pred_ctr.shape[1]
	pred_ctr = pred_ctr.to(torch.float64)
	pred_ctr = pred_ctr[:, :] * torch.cuda.DoubleTensor([w, h], device=img.device)
	shape_loss = torch.zeros(img.shape[0], dtype=torch.float32, device=img.device)
	# k = 100.0
	# w = (k**2)/(pt_num*pt_num)
	w = 1.0 / (8 * np.pi)
	inv_l = sample_tps_inv_l[:-3, :-3]
	for i in range(0, n):
		pred_x, pred_y = pred_ctr[i][:, 0].unsqueeze(0), pred_ctr[i][:, 1].unsqueeze(0)
		bending_energy = torch.mm(torch.mm(pred_x, inv_l), pred_x.t()) + torch.mm(torch.mm(pred_y, inv_l), pred_y.t())
		shape_loss[i] = max(w * bending_energy, 0)
	return shape_loss.mean()


def get_tps_shape_loss(img, pred_ctr, sample_tps_l, sample_tps_inv_l):
	[n, c, h, w] = img.shape
	pt_num = pred_ctr.shape[1]
	pred_ctr = pred_ctr.to(torch.float64)
	pred_ctr = pred_ctr[:, :] * torch.cuda.DoubleTensor([w, h], device=img.device)
	shape_loss = torch.zeros(img.shape[0], dtype=torch.float32, device=img.device)
	for i in range(0, n):
		V = torch.cat((torch.transpose(pred_ctr[i], 0, 1), torch.zeros((2, 3), dtype=torch.float64, device=img.device)), 1)
		Wa = torch.mm(sample_tps_inv_l, torch.transpose(V, 0, 1))
		# VT = torch.transpose(V,0,1)
		# for i in range(0, VT.shape[0]):
		#     print(torch.mm(tmp, Wa)[i], VT[i], torch.mm(tmp, Wa)[i] - VT[i])

		W = Wa[:-3, :]
		a = Wa[-3:, :]
		K = sample_tps_l[:-3, :-3]

		WK = torch.mm(torch.transpose(W, 0, 1), K)
		WKW = torch.mm(WK, W)
		shape_loss[i] = max(0.5 * torch.trace(WKW), 0)

	return shape_loss
