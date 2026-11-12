"""Compose measured RGB/depth panels; no generated or estimated imagery."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json,shutil
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'input_candidates/B_meeting_table/source.json'
m=json.loads(SOURCE.read_text())
# Viridis-style sequential palette, common metric range across all three views.
anchors=[(68,1,84),(59,82,139),(33,145,140),(94,201,98),(253,231,37)]
def color(value):
 if value<=0:return (0,0,0)
 t=min(1,max(0,value/7000))*4;k=min(3,int(t));f=t-k
 return tuple(round(anchors[k][c]*(1-f)+anchors[k+1][c]*f) for c in range(3))
w,h=1920,1440;gap=32
clean=Image.new('RGB',(3*w+2*gap,2*h+gap),'white')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',56)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',42)
for j,v in enumerate(m['views']):
 rgb=Image.open(v['image']).convert('RGB');assert rgb.size==(w,h)
 dep=Image.open(v['depth']);assert dep.mode=='I;16'
 shutil.copy2(v['depth'],OUT/f"depth_view{j+1}_{v['frame']}_raw16.png")
 vis=Image.new('RGB',dep.size);vis.putdata([color(x) for x in dep.getdata()])
 vis.save(OUT/f'depth_view{j+1}_color.png')
 clean.paste(rgb,(j*(w+gap),0));clean.paste(vis.resize((w,h),Image.Resampling.NEAREST),(j*(w+gap),h+gap))
clean.save(OUT/'B_RGBD_clean.png',dpi=(300,300))
clean.save(OUT/'B_RGBD_clean.pdf',resolution=300)
# Labeled preview with shared metric scale below the complete panel.
margin=170;top=110;bottom=190
labeled=Image.new('RGB',(clean.width+margin,clean.height+top+bottom),'white');labeled.paste(clean,(margin,top));d=ImageDraw.Draw(labeled)
for j in range(3):d.text((margin+j*(w+gap)+w//2,20),f'View {j+1}',font=font,fill='black',anchor='mt')
d.text((15,top+h//2),'RGB',font=small,fill='black');d.text((15,top+h+gap+h//2),'Depth',font=small,fill='black')
x0=margin+700;y0=top+clean.height+45;bw=2800;bh=45
for x in range(bw):d.line((x0+x,y0,x0+x,y0+bh),fill=color(max(1,7000*x/(bw-1))))
for k in range(8):
 x=x0+int(bw*k/7);d.text((x,y0+bh+12),str(k),font=small,fill='black',anchor='mt')
d.text((x0+bw+70,y0),'m  (near → far)',font=small,fill='black')
labeled.save(OUT/'B_RGBD_labeled.png',dpi=(300,300))
preview=labeled.copy();preview.thumbnail((1800,1200));preview.save(OUT/'B_RGBD_preview.jpg',quality=95)
m['visualization']={'depth_units':'millimeters in raw PNG; meters in legend','range_m':[0,7],'palette':'interpolated viridis-style sequential palette; near purple, far yellow','invalid_depth':'black','depth_original_resolution':[256,192],'rgb_resolution':[1920,1440],'depth_display_resampling':'nearest-neighbor; no reprojection/registration applied','layout':'RGB top, corresponding depth bottom; three views left to right'}
(OUT/'provenance.json').write_text(json.dumps(m,indent=2)+'\n')
(OUT/'README.md').write_text('''# B 组 RGB-D 输入拼图

上排为三张 RGB，下排为各自对应的实测深度图，列顺序为 6930、7520、14130。

- `B_RGBD_clean.png` / `.pdf`：无标签、无色条，适合嵌入主架构图。
- `B_RGBD_labeled.png`：包含视图、模态标签及统一深度色条。
- `B_RGBD_preview.jpg`：便于查看的小尺寸预览。
- `depth_view*_raw16.png`：原始 16 位深度，未修改。
- `depth_view*_color.png`：原分辨率深度可视化。

三张深度采用统一 0–7 m 色标，紫近黄远，无效深度为黑色。原深度为 256×192，拼图仅以最近邻方式放大到 RGB 显示尺寸，不宣称执行了 RGB-D 几何重投影或精确像素配准。RGB 保持原始取景。深度毫米转米与本地数据加载实现一致。
''')
print(clean.size, labeled.size)
