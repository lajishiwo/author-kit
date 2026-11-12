# Part 1 [LaTeX]

```latex
\section{Introduction}
\label{sec:intro}
Spatial reasoning requires relating visual observations to quantities and relations in a three-dimensional scene. Relevant tasks include estimating distances, matching objects across views, and determining how spatial relations change with viewpoint. SPAR evaluates these capabilities across spatial tasks and cognitive levels~\cite{spar}. These tasks motivate a joint examination of the evidence available to a vision-language model and the mechanism through which that evidence supports reasoning.

3DThinker investigates geometric imagination and continuous latent reasoning from limited views~\cite{thinker}. We initialize PIGeR from the pretrained 3DThinker-Mindcube checkpoint and retain the eight-step recurrent latent computation used in our experimental configuration. Our central question is whether changing the evidence supplied to this computation can improve its effectiveness: \emph{can measured camera poses and explicit RGB-D representations make continuous spatial reasoning more effective?} We investigate externally grounded geometric inputs while preserving the underlying latent reasoning mechanism.

PIGeR (Pose-Integrated Geometric Reasoning) integrates measured relative poses and frozen BIP3D RGB-D features into the existing visual prefix~\cite{bip3d}. We adapt CGMF from SpaceMind~\cite{spacemind} to use camera embeddings for geometric K/V conditioning and output gating. Fusion operates after the frozen Qwen visual merger and before the resulting embeddings replace the image placeholders in the language input. The number, order, and dimensions of the visual tokens remain unchanged. A residual scale initialized to zero preserves the original visual prefix at initialization. Language-side LoRA adapters~\cite{lora} and the fusion branch are then optimized jointly with the original assistant-sequence objective.

The contribution lies in integrating and evaluating explicit geometric evidence within an existing continuous-reasoning system. PIGeR reuses the latent mechanism of 3DThinker, the feature extractor of BIP3D, and the fusion design of SpaceMind. Question text is supplied to the language decoder, whereas neither CGMF nor geometric feature sampling receives it directly. Official task markers remain part of the RGB inputs and can therefore provide indirect target cues to both visual branches. Geometry affects the continuous latents through the enhanced visual prefix. During answer generation, the decoder can attend to both the prefix and latent states, so latent changes alone do not establish that geometric information improves reasoning through the latent pathway.

Our experimental design separates the effect of additional inputs from the effect of camera-specific conditioning. GENERIC receives the same RGB-D inputs, calibration, and camera embeddings as CGMF, with the same trainable parameter count and initialization. It replaces the camera-specific conditioning sources in the K/V bias and output gate while retaining camera embeddings in attention memory. The comparison therefore evaluates these conditioning choices jointly. B0 provides a historical RGB-only reference at a matched optimization budget, with the additional inputs and parameters of the fusion variants explicitly acknowledged.

The verified 10K results show mixed effects. CGMF achieves a higher Full macro score than GENERIC, but both remain below B0. The paired confidence intervals for CGMF relative to either comparator include zero. Improvements on some metric and viewpoint tasks coexist with decreases on multi-view relation and imagination tasks. These observations motivate controlled interventions and later-checkpoint evaluations; they do not establish an overall improvement in spatial reasoning.

This study provides an explicit integration of measured pose and frozen RGB-D features that preserves the visual-token interface, a comparison with matched inputs and trainable parameter counts that isolates camera-specific conditioning, and an auditable interim evaluation with task-level analysis and a defined intervention protocol. We distinguish implementation properties, observed results, and hypotheses that require further experimental validation.
```

```latex
\section{Related Work}
\label{sec:related}
\paragraph{Continuous spatial reasoning.}
3DThinker investigates geometric imagination for spatial reasoning from limited views~\cite{thinker}. We retain its pretrained backbone and use eight recurrent continuous latents in the present configuration. PIGeR modifies the visual evidence available to the recurrent computation while preserving the recurrence itself. Because answer generation can also attend directly to the visual prefix, any benefit of geometric inputs must be distinguished from a demonstrated effect mediated by the latent states.

\paragraph{Camera-guided multimodal fusion.}
SpaceMind introduces CGMF to guide fusion between visual and spatial features using camera representations~\cite{spacemind}. PIGeR reuses its camera-conditioned K/V feature bias, geometry importance weighting, and output gating. The adaptation connects measured relative poses and frozen BIP3D features to the post-merger visual interface of 3DThinker. No pretrained SpaceMind CGMF weights are loaded. Our comparison with GENERIC evaluates camera-specific conditioning under matched inputs and trainable parameter counts; it does not treat CGMF as a new algorithm.

\paragraph{Explicit RGB-D features.}
BIP3D connects image representations with 3D perception for embodied intelligence~\cite{bip3d}. We reuse the frozen feature extraction components of its Tiny Det checkpoint, including RGB and depth feature extraction and spatial enhancement. The resulting features combine appearance, depth, and position information; they are neither object detections nor an explicit scene mesh. Relative rotation is encoded using the first two columns of a rotation matrix, following the continuous 6D representation studied by Zhou \etal~\cite{rotation}. This representation supplies a pose input without imposing equivariance on the learned fusion module.

\paragraph{Adaptation and evaluation.}
LoRA learns low-rank parameter updates while keeping the underlying weights frozen~\cite{lora}. We apply it to the language decoder and jointly train the added fusion components. SPAR provides spatial perception and reasoning tasks with RGB-D observations and camera metadata~\cite{spar}. Our experiments use the local 20-category configuration and report the task macro average. The task-specific scores are not uniformly binary accuracies. Comparisons with RGB-only B0 account explicitly for the measured depth and calibration available to PIGeR and GENERIC.
```

```latex
\section{Method}
\label{sec:method}
\subsection{Inputs and Fusion Interface}
For a sample with $S$ views, the inputs comprise RGB images $I_i$, measured depth $D_i$, intrinsics $K_i$, camera-to-world transforms $T_{w\leftarrow i}$, and a question $q$. We distinguish the color and depth intrinsics during registration. Training and development images include the official task markers; public evaluation images already contain them. Both visual branches receive the same marked source images, with separate resizing and normalization. The frozen Qwen visual encoder and merger produce $V_i\in\mathbb{R}^{n_i\times2048}$. A geometric branch produces $G_i\in\mathbb{R}^{16\times2048}$ and one camera embedding $P_i\in\mathbb{R}^{2048}$ per view. Fusion returns $V'_i$ of exactly the same shape as $V_i$.

The language model receives the question and the enhanced visual prefix. The enhanced embeddings replace the existing image placeholders; camera and geometry embeddings are not appended to the language input. CGMF and geometric feature sampling do not directly receive question text, although image markers can provide indirect target cues. Invalid geometric entries are masked, and fusion is bypassed when no valid geometry is available. Figure~\ref{fig:architecture} summarizes the interface.

\begin{figure}[t]
\centering
\fbox{\parbox{0.92\columnwidth}{\small
\textbf{Architecture interface.}\\
RGB $\rightarrow$ frozen Qwen encoder/merger $\rightarrow V$.\\
RGB-D and calibration $\rightarrow$ frozen BIP3D $\rightarrow G$.\\
Relative pose $\rightarrow$ pose projection $\rightarrow P$.\\
$(V,G,P)$ $\rightarrow$ adapted CGMF and residual scale $\rightarrow V'$.\\
$(V',q)$ $\rightarrow$ language model with LoRA $\rightarrow$ eight continuous latents $\rightarrow$ answer.
}}
\caption{PIGeR data flow. Geometric fusion updates existing visual tokens before language processing without adding language input tokens. The same language decoder performs latent recurrence and answer generation; the visual and question prefix remains accessible during decoding.}
\label{fig:architecture}
\end{figure}

\subsection{Relative Pose and Coordinate Conventions}
We anchor the pose branch to the first view and compute
\begin{equation}
T_{i\leftarrow0}=T_{w\leftarrow i}^{-1}T_{w\leftarrow0}.
\label{eq:pose}
\end{equation}
This transform maps coordinates from the reference camera to camera $i$. With rotation $R_{i\leftarrow0}$ and translation $t_{i\leftarrow0}$, we form
\begin{equation}
p_i=[\operatorname{R6D}(R_{i\leftarrow0});t_{i\leftarrow0}],\quad
P_i=W_p p_i+b_p,
\end{equation}
where $p_i\in\mathbb{R}^9$, translation is in meters, and $W_p$ maps 9 to 2048 dimensions. R6D uses the first two rotation columns~\cite{rotation}. The reference transform is the identity in exact arithmetic, with possible small residuals from floating-point inversion and multiplication. The translation in Eq.~\ref{eq:pose} is expressed within the coordinate transform and does not directly represent camera displacement in world coordinates.

The BIP3D branch uses $T_{0\leftarrow i}=T_{w\leftarrow0}^{-1}T_{w\leftarrow i}$ to express each view in reference-camera coordinates. Its internal projection applies the inverse of this transform, together with the adjusted intrinsics, to project reference-frame points into the current view. The two relative transforms therefore serve complementary roles. They derive from the same calibration: BIP3D features already depend on camera pose, so removing only the direct pose embeddings does not remove all camera information.

\subsection{Frozen Local RGB-D Representation}
Measured depth is converted from millimeters to meters, back-projected using the depth intrinsics, and projected into the color-camera grid using the color intrinsics. The registered grid is then resized to the depth resolution, with nearest-pixel assignment and a z-buffer retaining the nearest valid depth. Uncovered pixels are set to zero. RGB is resized bilinearly to the same dimensions, and the color intrinsics are adjusted accordingly. The implementation retains valid depths of approximately 0.2--10 meters and assumes rectified pinhole inputs with coincident sensor coordinate frames. The training and development loader rejects nonidentity color/depth sensor extrinsics; arbitrary uncalibrated RGB-depth sensor pairs are outside this input convention.

The frozen BIP3D Tiny Det extractor~\cite{bip3d} uses an RGB backbone and neck, a depth backbone and neck, and a spatial enhancement module. The detection decoder, text encoder, and text-image enhancement modules are not used. RGB features are extracted with Swin-T and mapped to four scales with 256 channels. A narrow two-dimensional ResNet34 processes the single-channel metric depth and produces four scales with 32 channels. This depth backbone does not operate on a point cloud.

At each feature location, the spatial enhancer constructs 64 candidate depths over 0.25--10 meters and transforms the corresponding points into the reference-camera frame. RGB and depth features predict softmax weights over these candidates. The weighted linear embeddings of candidate positions are concatenated with RGB and depth features, projected from 320 to 256 dimensions, combined with an RGB residual, and normalized. Consequently, the output combines appearance, depth, and positional information. Neither the candidate-depth distribution nor a sequence of ground-truth XYZ coordinates is passed separately to CGMF. Feature enhancement is performed per view in a common reference frame; cross-view object association through detection decoding is not part of this feature extraction path.

The implementation samples 48 local features per scene using a fixed image grid: 48 for one view, 24 per view for two views, and 16 per view for three views. Grid positions are aligned with nearby valid-depth projections when available. Features sampled bilinearly from the four scales are averaged.

Before fusion, 16 features per view are selected deterministically and projected from 256 to 2048 dimensions. The total geometry memory therefore contains 16, 32, or 48 tokens for one, two, or three views, respectively. The cached features are bound to the ordered RGB, depth, intrinsics, extrinsics, and extraction configuration. Sampling uses neither question text nor object identities or boxes. Raw XYZ coordinates, raw depth, boxes, answers, and relation labels are not concatenated as additional geometry tokens. This compact, question-independent sampling can omit details needed to bind geometric evidence to a target object.

\subsection{Adapted Camera-Guided Fusion}
We adapt CGMF from SpaceMind~\cite{spacemind} using visual, pose, and geometry widths of 2048 and an internal attention width of $d=256$. Attention uses four heads of 64 dimensions, and fusion dropout is set to zero. The following equations summarize the computational roles; learned linear maps include their implementation biases.

\paragraph{Camera-conditioned keys and values.}
Let $\bar V_i=\operatorname{LN}(V_i)$ and $\bar G_i=\operatorname{LN}(G_i)$. Initial projections are
\begin{equation}
Q_i=\bar V_iW_Q,\quad K_i^g=\bar G_iW_K,\quad U_i^g=\bar G_iW_U.
\end{equation}
Project the current camera as $c_i=W_cP_i+b_c$. Broadcasting $c_i$ over the 16 tokens gives the feature bias
\begin{equation}
B_i=\operatorname{MLP}([\bar G_i;\operatorname{repeat}_{16}(c_i)]).
\end{equation}
The same bias is added to $K_i^g$ and $U_i^g$. It conditions the feature representations; it does not impose an attention-logit constraint computed directly from metric distance or projection error.

\paragraph{Geometry importance.}
A geometry-only self-attention block produces context $C_i$. A learned residual projection and LayerScale, initialized to a scale of 0.00001, combine this context with $\bar G_i$ to predict sigmoid weights:
\begin{equation}
w_i=\sigma\!\left(\operatorname{MLP}_w(W_r\bar G_i+\operatorname{LS}(C_i))\right).
\end{equation}
The weighted values are $\widetilde U_i^g=w_i\odot(U_i^g+B_i)$ and keys are $\widetilde K_i^g=K_i^g+B_i$. These weights represent learned token importance and are not calibrated depth-confidence estimates with separate supervision.

\paragraph{Local geometry and all-view cameras.}
For view $i$, define camera memory rows $M^c_{i,j}=c_j+e_j+\mathbf{1}[i=j]e_{\rm cur}$, with sinusoidal view-order encoding $e_j$ and a learned current-view marker. Cross-attention uses
\begin{align}
K_i^*&=[M_i^c;\widetilde K_i^g],\quad U_i^*=[M_i^c;\widetilde U_i^g],\\
H_i&=\operatorname{LN}\!\left(W_o\operatorname{Attn}(Q_i,K_i^*,U_i^*)\right).
\end{align}
Each visual query therefore attends to the 16 geometry tokens of its own view and all $S$ camera tokens. Geometry tokens from other views are not directly included in this memory. Camera tokens receive view-order encodings, whereas geometry tokens receive no additional encoding of their array indices; their feature values already contain spatial information from BIP3D. Cross-view integration can subsequently occur in the shared language context.

\paragraph{Gating and residual integration.}
Camera-conditioned SwiGLU computes the visual update using $W_hH_i\odot\operatorname{SiLU}(W_gc_i+b_g)$. Let $F_{\rm CGMF}$ denote the complete reused block, including its internal visual residual and dropout. The outer integration is
\begin{equation}
V'_i=V_i+\alpha\big(F_{\rm CGMF}(V_i,P,G)-V_i\big),\quad\alpha_0=0.
\label{eq:residual}
\end{equation}
At initialization, the fused prefix equals the original visual prefix. The scalar $\alpha$ is unconstrained and can become negative; its sign alone does not indicate whether the branch is useful. Token importance weights, the camera-conditioned channel gate, and the outer residual scale serve distinct roles. At zero residual scale, the scale can receive a gradient before the remaining fusion parameters receive effective QA gradients. The implementation retains the internal residual addition followed by subtraction and outer scaling, so BF16 rounding can differ from an algebraically simplified update. Pose and geometry projections and CGMF are initialized without pretrained SpaceMind fusion weights.

\subsection{Continuous Latents and Training}
The language decoder uses the final hidden state from the preceding computation as the input embedding at the next latent position, repeating this sequential operation eight times. The latent states are generated recurrently rather than implemented as eight independent learnable queries. In schematic notation,
\begin{equation}
Z=F_{\rm latent}(V',q),\qquad Z=(z_1,\ldots,z_8),
\end{equation}
followed by autoregressive answer generation conditioned on $V'$, $q$, and $Z$ using the same decoder and LM head. The decoder has 36 layers, a hidden width of 2048, and grouped-query attention with 16 query heads and two K/V heads. Its causal attention retains access to the visual/question prefix and latent-related K/V states. Geometry can therefore affect answers both directly through the enhanced prefix $V'$ and indirectly through the recurrent latents. The latents have no prescribed camera, scene, or relation roles, and their intermediate geometric content is not explicitly supervised. Changes in $Z$ alone do not establish a causal contribution to answer quality.

Training starts from the original pretrained 3DThinker-Mindcube checkpoint. The Qwen visual encoder and merger, base language model, and BIP3D extractor remain frozen. The trainable components comprise language LoRA, pose and geometry projections, CGMF, and residual scaling: 29,933,568 LoRA parameters and 10,677,890 fusion-branch parameters, totaling 40,611,458. Freezing the base language weights preserves gradient propagation through its input activations, allowing the assistant loss to train the fusion branch.

The frozen Qwen visual branch uses a 32-layer ViT and a merger that combines neighboring patches in groups of four and projects them to width 2048. Dynamic resizing determines the visual-token count for each view; the count is not fixed at 256. Bridge padding is removed after fusion, preserving the actual token lengths, view order, and position indices. Image and text positions are handled by the original M-RoPE mechanism, which does not encode measured camera pose or metric object coordinates.

For supervised assistant positions $\mathcal A$, the objective remains
\begin{equation}
\mathcal L=-\sum_{t\in\mathcal A}\log p_\theta(y_t\mid y_{<t},V',q).
\end{equation}
The supervised assistant sequence includes latent-boundary and latent-placeholder positions, format tokens, and the answer. During training, latent inputs are sequentially replaced with the preceding hidden states while retaining the computation graph. During generation, the controller performs eight continuous updates before returning to vocabulary-based decoding. User-prefix and padding positions are masked in the objective. No geometric regression, distillation, contrastive objective, or reinforcement learning term is added; internal stage names do not change this supervised objective.

\subsection{Matched GENERIC Control}
GENERIC retains the same inputs, projections, token counts, trainable parameter count, and initialization. It changes two conditioning sources: the K/V bias MLP receives the geometry token and its key representation, and the output gate receives the mean visual query instead of the current camera projection. All-view camera tokens remain available as attention memory. Moreover, the BIP3D features retain their calibration-dependent spatial content. The comparison therefore isolates the two camera-specific conditioning choices jointly, rather than the presence or absence of camera information.
```

# Part 2 [Translation]
# 对应中文直译

## 1 引言

空间推理要求将视觉观测与三维场景中的数量和关系联系起来。相关任务包括估计距离、跨视图匹配物体，以及判断空间关系如何随视点变化。SPAR 从不同空间任务和认知层级评估这些能力。这些任务促使我们同时考察视觉语言模型可获得的证据，以及这些证据支持推理的机制。

3DThinker 研究从有限视图进行几何想象与连续潜在推理。我们从预训练的 3DThinker-Mindcube 检查点初始化 PIGeR，并保留当前实验配置中的八步递归潜在计算。我们的核心问题是：改变提供给这一计算过程的证据，是否能够提高其有效性，即实测相机位姿和显式 RGB-D 表示能否使连续空间推理更加有效？我们在保留底层潜在推理机制的同时，研究由外部观测提供依据的几何输入。

PIGeR 将实测相对位姿与冻结的 BIP3D RGB-D 特征整合到现有视觉前缀中。我们改编 SpaceMind 的 CGMF，利用相机嵌入调节几何 K/V 和输出门控。融合发生在冻结的 Qwen 视觉合并器之后，以及所得嵌入替换语言输入中的图像占位符之前。视觉词元的数量、顺序和维度保持不变。初始化为零的残差系数使初始化时的视觉前缀保持原样。随后，语言侧 LoRA 适配器与融合分支通过原有助手序列目标进行联合优化。

本研究的贡献在于将显式几何证据整合到已有连续推理系统中，并对此进行评估。PIGeR 复用 3DThinker 的潜在机制、BIP3D 的特征提取器和 SpaceMind 的融合设计。问题文本被提供给语言解码器，而 CGMF 与几何特征采样均不直接接收该文本。官方任务标记仍然包含在 RGB 输入中，因此能够为两个视觉分支提供间接的目标提示。几何信息通过增强后的视觉前缀影响连续潜在状态。在生成答案时，解码器能够同时关注前缀和潜在状态，因此，仅凭潜在状态发生变化，并不能证明几何信息通过潜在路径改善了推理。

我们的实验设计区分额外输入的影响与相机专用条件化的影响。GENERIC 与 CGMF 接收相同的 RGB-D 输入、标定信息和相机嵌入，并采用相同的可训练参数数量与初始化。它替换 K/V 偏置和输出门控中的相机专用条件来源，同时将相机嵌入保留在注意力记忆中。因此，该比较联合评估这两项条件化选择。B0 在匹配的优化预算下提供历史 RGB 参照，融合变体额外使用的输入和参数均被明确说明。

已核验的 10K 结果表现出混合效应。CGMF 的 Full 宏平均得分高于 GENERIC，但两者均低于 B0。CGMF 相对两个对照的配对置信区间均包含零。部分度量任务和视点任务的改善，与多视图关系任务和想象任务的下降同时出现。这些观测支持开展受控干预和后续检查点评估，但尚未建立空间推理整体改善的证据。

本研究提供了保留视觉词元接口的实测位姿与冻结 RGB-D 特征整合方式、通过匹配输入与可训练参数数量来分离相机专用条件化作用的比较，以及包含任务级分析与明确干预方案的可审计阶段性评估。我们区分实现属性、观测结果和仍需进一步实验验证的假设。

## 2 相关工作

### 连续空间推理

3DThinker 研究从有限视图进行面向空间推理的几何想象。我们保留其预训练骨干，并在当前配置中使用八个递归连续潜在状态。PIGeR 修改递归计算可获得的视觉证据，同时保留递归机制本身。由于答案生成也能够直接关注视觉前缀，因此必须区分几何输入带来的收益与已经证明由潜在状态中介产生的效果。

### 相机引导的多模态融合

SpaceMind 提出 CGMF，利用相机表示引导视觉特征与空间特征的融合。PIGeR 复用其相机条件化 K/V 特征偏置、几何重要性加权和输出门控。该改编将实测相对位姿与冻结 BIP3D 特征接入 3DThinker 的合并器后视觉接口。我们不加载预训练的 SpaceMind CGMF 权重。与 GENERIC 的比较在匹配输入和可训练参数数量的条件下评估相机专用条件化，不将 CGMF 视为新算法。

### 显式 RGB-D 特征

BIP3D 将图像表示与面向具身智能的三维感知联系起来。我们复用其 Tiny Det 检查点中的冻结特征提取组件，包括 RGB 特征提取、深度特征提取和空间增强。所得特征结合外观、深度与位置信息，既不是物体检测结果，也不是显式场景网格。按照 Zhou 等人研究的连续六维表示，我们使用旋转矩阵的前两列编码相对旋转。该表示提供位姿输入，但不对学习到的融合模块施加等变性约束。

### 适配与评估

LoRA 在保持底层权重冻结的同时学习低秩参数更新。我们将其应用于语言解码器，并联合训练新增融合组件。SPAR 提供包含 RGB-D 观测和相机元数据的空间感知与推理任务。我们的实验使用本地二十类别配置，并报告任务宏平均。各任务得分并不统一对应二值准确率。在与仅使用 RGB 的 B0 比较时，我们明确说明 PIGeR 与 GENERIC 可获得的实测深度和标定信息。

## 3 方法

### 输入与融合接口

对于包含 $S$ 个视图的样本，输入包括 RGB 图像 $I_i$、实测深度 $D_i$、内参 $K_i$、相机到世界变换 $T_{w\leftarrow i}$，以及问题 $q$。在配准过程中，我们区分彩色内参与深度内参。训练和开发图像包含官方任务标记；公开评测图像已经包含这些标记。两个视觉分支接收相同的带标记源图像，并分别进行缩放与归一化。冻结的 Qwen 视觉编码器和合并器生成 $V_i\in\mathbb{R}^{n_i\times2048}$。几何分支为每个视图生成 $G_i\in\mathbb{R}^{16\times2048}$ 和一个相机嵌入 $P_i\in\mathbb{R}^{2048}$。融合输出 $V'_i$ 与 $V_i$ 的形状完全一致。

语言模型接收问题和增强后的视觉前缀。增强嵌入替换现有图像占位符；相机嵌入与几何嵌入不被附加到语言输入中。CGMF 和几何特征采样不直接接收问题文本，但图像标记能够提供间接目标提示。无效几何项被掩蔽；没有有效几何时，融合被旁路。架构图概括了这一接口。

图示说明：几何融合在语言处理之前更新现有视觉词元，不增加语言输入词元。同一语言解码器执行潜在递归与答案生成；解码过程中仍可访问视觉和问题前缀。

### 相对位姿与坐标约定

我们以第一视图为位姿分支的参考，并计算 $T_{i\leftarrow0}=T_{w\leftarrow i}^{-1}T_{w\leftarrow0}$。该变换将参考相机坐标映射到相机 $i$。我们将旋转 $R_{i\leftarrow0}$ 的六维表示与平移 $t_{i\leftarrow0}$ 拼接，并通过线性映射生成相机嵌入，其中 $p_i\in\mathbb{R}^9$，平移单位为米，$W_p$ 将九维映射到 2048 维。R6D 使用旋转矩阵的前两列。在精确算术下，参考变换为恒等变换；浮点求逆和矩阵乘法可能引入微小残差。该平移在坐标变换内部表达，并不直接表示世界坐标中的相机位移。

BIP3D 分支使用 $T_{0\leftarrow i}=T_{w\leftarrow0}^{-1}T_{w\leftarrow i}$，将每个视图表达在参考相机坐标系中。其内部投影将该变换的逆与调整后的内参结合，将参考系中的点投影到当前视图。因此，这两个相对变换发挥互补作用。它们来自相同标定信息：BIP3D 特征已经依赖相机位姿，因此，仅移除直接位姿嵌入并不会移除全部相机信息。

### 冻结的局部 RGB-D 表示

实测深度先从毫米转换为米，使用深度内参反投影，再使用彩色内参投影到彩色相机栅格。随后，配准栅格被缩放到深度分辨率，采用最近像素分配，并通过深度缓冲保留最近的有效深度。未覆盖像素置零。RGB 通过双线性插值缩放到相同尺寸，彩色内参随之调整。当前实现保留约 0.2–10 米的有效深度，并假设输入满足校正后的针孔成像模型，且传感器坐标系重合。训练和开发加载器拒绝非恒等的彩色与深度传感器外参；任意未经标定的 RGB 与深度传感器组合不符合这一输入约定。

冻结的 BIP3D Tiny Det 提取器使用 RGB 骨干与颈部网络、深度骨干与颈部网络，以及空间增强模块。不使用检测解码器、文本编码器或文本图像增强模块。RGB 特征通过 Swin-T 提取，并映射到四个尺度，每个尺度具有 256 个通道。一个窄二维 ResNet34 处理单通道米制深度，并生成四个尺度、每尺度 32 个通道的特征。该深度骨干不处理点云。

在每个特征位置，空间增强器在 0.25–10 米范围内构造 64 个候选深度，并将相应点变换到参考相机坐标系。RGB 与深度特征预测这些候选点上的 softmax 权重。候选位置的加权线性嵌入与 RGB 和深度特征拼接，从 320 维投影到 256 维，与 RGB 残差结合后归一化。因此，输出结合外观、深度和位置信息。候选深度分布以及真实 XYZ 坐标序列均不会单独传递给 CGMF。特征增强在共同参考系中逐视图执行；这一特征提取路径不包含通过检测解码建立跨视图物体关联的过程。

当前实现使用固定图像网格为每个场景采样 48 个局部特征：单视图为 48 个，双视图为每视图 24 个，三视图为每视图 16 个。在存在邻近有效深度投影时，网格位置向其对齐。对四个尺度进行双线性采样后，所得特征取平均。

融合前，每视图确定性选择 16 个特征，并从 256 维投影到 2048 维。因此，一、二、三个视图对应的几何记忆总量分别为 16、32、48 个词元。缓存特征与有序 RGB、深度、内参、外参及提取配置绑定。采样既不使用问题文本，也不使用物体身份或框。原始 XYZ 坐标、原始深度、框、答案和关系标签均不作为附加几何词元拼接。紧凑且不依赖问题的采样可能遗漏将几何证据与目标物体绑定所需的细节。

### 改编的相机引导融合

我们改编 SpaceMind 的 CGMF，视觉、位姿和几何宽度均为 2048，内部注意力宽度为 $d=256$。注意力采用四个头，每头 64 维，融合中的丢弃率设为零。原有公式概括各项计算作用，学习得到的线性映射包含实现中的偏置。

相机条件化的键和值：首先对视觉与几何特征归一化，并计算查询、键和值投影。当前相机被投影为 $c_i=W_cP_i+b_c$。将其广播到 16 个词元后，与几何特征共同生成特征偏置。同一偏置被添加到 $K_i^g$ 和 $U_i^g$。它对特征表示进行条件化，而不是施加由米制距离或投影误差直接计算的注意力分数约束。

几何重要性：仅使用几何的自注意力模块生成上下文 $C_i$。可学习残差投影和初始化尺度为 0.00001 的 LayerScale 将该上下文与 $\bar G_i$ 结合，以预测 sigmoid 权重。加权值为 $\widetilde U_i^g=w_i\odot(U_i^g+B_i)$，键为 $\widetilde K_i^g=K_i^g+B_i$。这些权重表示学习得到的词元重要性，不是具有独立监督的校准深度置信度估计。

局部几何与全部视图的相机：对于视图 $i$，相机记忆包含各相机投影、正弦视图顺序编码和可学习的当前视图标记。每个视觉查询关注本视图的 16 个几何词元和全部 $S$ 个相机词元。其他视图的几何词元不直接包含在该记忆中。相机词元接收视图顺序编码，而几何词元不额外接收数组索引编码；其特征值已经包含 BIP3D 提供的空间信息。随后可在共享语言上下文中进行跨视图整合。

门控与残差整合：相机条件化 SwiGLU 使用 $W_hH_i\odot\operatorname{SiLU}(W_gc_i+b_g)$ 计算视觉更新。$F_{\rm CGMF}$ 表示完整复用模块，包括内部视觉残差和丢弃操作。外层按照保留的残差公式整合，且残差系数初始化为零。初始化时，融合前缀与原始视觉前缀相同。标量 $\alpha$ 不受约束，可以变为负值；其符号本身不能说明该分支是否有效。词元重要性权重、相机条件化通道门和外层残差系数承担不同作用。残差系数为零时，该系数可以先获得梯度，其余融合参数随后才能获得有效 QA 梯度。实现保留内部残差相加、相减和外层缩放，因此 BF16 舍入结果可能不同于代数简化后的更新。位姿投影、几何投影和 CGMF 的初始化不使用预训练的 SpaceMind 融合权重。

### 连续潜在状态与训练

语言解码器将前一次计算的最终隐藏状态作为下一潜在位置的输入嵌入，顺序重复该操作八次。潜在状态递归生成，而不是实现为八个相互独立的可学习查询。计算得到 $Z=F_{\rm latent}(V',q)$，其中 $Z=(z_1,\ldots,z_8)$，随后使用同一解码器和语言模型输出头，在 $V'$、$q$、$Z$ 的条件下自回归生成答案。解码器具有 36 层、2048 维隐藏宽度，分组查询注意力包含 16 个查询头和两个 K/V 头。其因果注意力持续访问视觉和问题前缀，以及与潜在状态相关的 K/V。因此，几何信息既可通过增强前缀 $V'$ 直接影响答案，也可通过递归潜在状态间接影响答案。潜在状态没有预设的相机、场景或关系角色，其中间几何内容不受显式监督。仅凭 $Z$ 的变化不能证明其对答案质量具有因果贡献。

训练从原始预训练 3DThinker-Mindcube 检查点开始。Qwen 视觉编码器和合并器、基础语言模型与 BIP3D 提取器保持冻结。可训练组件包括语言 LoRA、位姿和几何投影、CGMF 与残差系数：LoRA 参数为 29,933,568 个，融合分支参数为 10,677,890 个，总计 40,611,458 个。冻结基础语言权重仍然保留经过输入激活的梯度传播，使助手损失能够训练融合分支。

冻结的 Qwen 视觉分支使用 32 层 ViT 和合并器，后者将相邻图像块以四个一组结合并投影到 2048 维。动态缩放决定每个视图的视觉词元数量，该数量并非固定为 256。融合后移除桥接过程中的填充，保留真实词元长度、视图顺序和位置索引。图像与文本位置由原有 M-RoPE 机制处理，该机制不编码实测相机位姿或米制物体坐标。

对于受监督的助手位置，目标仍为原有自回归交叉熵。受监督助手序列包括潜在边界位置、潜在占位位置、格式词元和答案。训练时，潜在输入依次由前一隐藏状态替换，并保留计算图。生成时，控制器执行八次连续更新，然后返回基于词表的解码。用户前缀和填充位置在目标中被掩蔽。不增加几何回归、蒸馏、对比目标或强化学习项；内部阶段名称不改变该监督目标。

### 匹配的 GENERIC 对照

GENERIC 保留相同输入、投影、词元数量、可训练参数数量与初始化。它改变两个条件来源：K/V 偏置 MLP 接收几何词元及其键表示，输出门接收视觉查询均值，而非当前相机投影。全部视图的相机词元仍可作为注意力记忆使用。此外，BIP3D 特征保留其依赖标定的空间内容。因此，该比较联合分离两项相机专用条件化选择的作用，而不是相机信息存在与否的作用。


# Part 3 [Modification Log]
仅修改引言、相关工作与方法三章，摘要、实验正文、结果表和结论保持原样。重写句式以增强逻辑衔接、正式性和可读性，消除所有格表达并保留原有缩写。根据结构审阅补充任务标记的间接提示、RGB-D 注册约定、互逆位姿路径、BIP3D 特征提取范围与位置增强、每视图几何词元数量、CGMF 条件化与门控，以及同一解码器中的连续递归和答案前缀旁路。保留原有公式、数学表达、引用键、交叉引用、标签及强调格式，未添加性能结论。 编译验证通过，当前 PDF 共七页，无未定义引用、版面溢出或字体替代。
