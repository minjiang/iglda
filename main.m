%% ITCA is the former name of IGLDA

clear all;
close all;
clc

%% Four domains: { Caltech10, amazon, webcam, dslr }
src = 'dslr';
tgt = 'webcam';
nPerClass = 8;

load(['data/' src '_SURF_L10.mat']);     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(fts,1);    clear fts
Ys = labels;           clear labels

load(['data/' tgt '_SURF_L10.mat']);     % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(fts,1);     clear fts
Yt = labels;            clear labels

%%
method = 'ITCA';
mu = 10;
lambda = .1;
dim = 300;
p1 = 1e-2;

DA(method, mu, lambda, dim, p1, Xs, Ys, Xt, Yt, nPerClass);