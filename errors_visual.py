# 상위 변수 시각화
top_vars = total_contrib.drop(columns='y6_pred').abs().mean().sort_values(ascending=False).head(10).index

plt.figure(figsize=(14, 6))
for var in top_vars:
    plt.plot(total_contrib.index, total_contrib[var], label=var)
plt.title("Top 10 Variable Contributions Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/top10_contrib_lineplot.png", dpi=300)
plt.show()
plt.close()

# 개별 subplot

# 색상 고정용 colormap 생성
colors = plt.cm.tab10(np.linspace(0, 1, len(top_vars))) # 또는 cm.Set3

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 14), sharex=True)
axes = axes.flatten()
for i, var in enumerate(top_vars):
    axes[i].plot(total_contrib.index, total_contrib[var], color=colors)
    axes[i].set_title(var)
    axes[i].grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/top10_contrib_subplots.png", dpi=300)
plt.show()
plt.close()
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[20], line 9
      7 axes = axes.flatten()
      8 for i, var in enumerate(top_vars):
----> 9     axes[i].plot(total_contrib.index, total_contrib[var], color=colors)
     10     axes[i].set_title(var)
     11     axes[i].grid(True)

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\matplotlib\axes\_axes.py:1777, in Axes.plot(self, scalex, scaley, data, *args, **kwargs)
   1534 """
   1535 Plot y versus x as lines and/or markers.
   1536 
   (...)
   1774 (``'green'``) or hex strings (``'#008000'``).
   1775 """
   1776 kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
-> 1777 lines = [*self._get_lines(self, *args, data=data, **kwargs)]
   1778 for line in lines:
   1779     self.add_line(line)

File c:\Users\SKSiltron\Desktop\project\signpost_project\signpost_project_env\Lib\site-packages\matplotlib\axes\_base.py:297, in _process_plot_var_args.__call__(self, axes, data, return_kwargs, *args, **kwargs)
    295     this += args[0],
    296     args = args[1:]
--> 297 yield from self._plot_args(
...
       [0.54901961, 0.3372549 , 0.29411765, 1.        ],
       [0.89019608, 0.46666667, 0.76078431, 1.        ],
       [0.49803922, 0.49803922, 0.49803922, 1.        ],
       [0.7372549 , 0.74117647, 0.13333333, 1.        ],
       [0.09019608, 0.74509804, 0.81176471, 1.        ]]) is not a valid value for color: supported inputs are (r, g, b) and (r, g, b, a) 0-1 float tuples; '#rrggbb', '#rrggbbaa', '#rgb', '#rgba' strings; named color strings; string reprs of 0-1 floats for grayscale values; 'C0', 'C1', ... strings for colors of the color cycle; and pairs combining one of the above with an alpha value
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
