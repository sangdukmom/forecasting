print(model_trend)

DLinear_with_emb(
  (decomp): series_decomp(
    (moving_avg): moving_avg(
      (avg): AvgPool1d(kernel_size=(3,), stride=(1,), padding=(0,))
    )
  )
  (Linear_Seaonal): Linear(in_features=12, out_features=1, bias=True)
  (Linear_Trend): Linear(in_features=12, out_features=1, bias=True)
  (fc1): Linear(in_features=247, out_features=1, bias=True)
  (fc2): Linear(in_features=247, out_features=1, bias=True)
  (periodic): PeriodicEmbeddings(
    (periodic): _Periodic()
    (linear): Linear(in_features=48, out_features=12, bias=True)
    (activation): ReLU()
  )
  (emb_linear): Linear(in_features=12, out_features=1, bias=True)

---------------------------------

for name, param in model_trend.named_parameters():
    print(name, param.shape)

Linear_Seaonal.weight torch.Size([1, 12])
Linear_Seaonal.bias torch.Size([1])
Linear_Trend.weight torch.Size([1, 12])
Linear_Trend.bias torch.Size([1])
fc1.weight torch.Size([1, 247])
fc1.bias torch.Size([1])
fc2.weight torch.Size([1, 247])
fc2.bias torch.Size([1])
periodic.periodic.wegiht torch.Size([247, 24])
periodic.linear.weight torch.Size([12, 48])
periodic.linear.bias torch.Size([12])
emb_linear.weight torch.Size([1, 12])
emb_linear.bias torch.Size([1])
