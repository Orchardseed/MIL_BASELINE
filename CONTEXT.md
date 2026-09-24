# MIL Research Domain

Canonical English terminology with Chinese term labels for research using precomputed foundation-model features. These definitions describe meaning; they do not assert that every current model supports every input or task.

## Language

### Project domain

**MIL Research Toolkit / MIL 科研工具库**:
A general research toolkit for downstream learning from precomputed foundation-model features across projects and data modalities. It comprises a reusable MIL model library and a general training and evaluation framework, designed to be independently usable and portable through explicit interfaces. External projects prepare inputs to satisfy its input contracts; feature production and project-specific scientific preprocessing remain outside the toolkit.
_Avoid_: equating the toolkit with one research project's complete pipeline.

### Features and research units

**Foundation Model / 基础模型**:
A pretrained model used to produce representations for downstream research. Its outputs can have different feature dimensions, structures, and semantic units.
_Avoid_: equating the feature-producing model with the downstream MIL model.

**Precomputed Features / 预计算特征**:
Representations saved before a downstream MIL run. They may already include upstream transformations or weighting; precomputed does not mean raw or unmodified.
_Avoid_: assuming precomputed means raw, unweighted, or unmodified.

**Research Sample / 研究样本**:
The unit to which a research target or prediction refers. Its relationship to a patient, case, slide, or feature bag is defined by the project protocol.
_Avoid_: assuming the prediction unit is always a patient, slide, or bag.

**Instance / 实例**:
An element represented within a MIL input, such as a WSI tile, an internal token, or a location or region in a two- or three-dimensional representation. Its semantic meaning is established by the input contract, not by tensor shape alone.
_Avoid_: assuming a WSI tile and an internal token are the same semantic unit.

**Feature Bag / 特征袋**:
A collection of related feature instances presented together as a MIL input, with corresponding information such as two- or three-dimensional positions when required. A bag is not automatically an independent biological case, and several bags are not automatically one prediction unit.
_Avoid_: equating a bag with a patient or case; assuming shared case membership requires bag merging.

**Feature Structure / 特征结构**:
The organization and meaning of a representation's axes, instances, sources, and relationships. A matrix, spatial grid, and hierarchy may encode different relationships even when they can be reshaped into similar arrays.
_Avoid_: treating reshape-compatible arrays as semantically equivalent.

**Feature Metadata / 特征元数据**:
Information needed to interpret feature instances, such as coordinates, validity, weights, source indices, scale, or grid geometry. Positional information refers to corresponding instances and has a defined coordinate system, axis order, and units where applicable. Grid indices and physical positions are distinct meanings. Metadata and feature values have distinct meanings.
_Avoid_: equating grid indices with physical coordinates; assuming stored coordinates are necessarily used by the model.

**MIL Input Contract / MIL 输入约定**:
The declared structure, semantic meaning, required accompanying information, and correspondence relationships under which an input can be consumed by the MIL toolkit. It describes the prepared input independently of the external project's data-preparation procedure. Conformance to a library input contract does not establish compatibility with every MIL model.
_Avoid_: treating file readability or a matching shape as complete input conformance.

**External Preprocessing / 外部预处理**:
Data preparation performed outside the MIL toolkit to produce inputs conforming to its declared contracts. The external project owns the scientific meaning of those transformations; the toolkit consumes the prepared result without taking ownership of their generation.
_Avoid_: using Input Adapter to name project-level data preparation.

**Canonical Value / 权威来源确定的规范值**:
A value whose interpretation and identity are established by its authoritative source. A downstream representation refers to that value without silently redefining it.
_Avoid_: silently redefining a value downstream.

### Models and tasks

**MIL Model / MIL 模型**:
A model that transforms, relates, and aggregates instances to produce task outputs. Input mapping, aggregation, and prediction are distinct responsibilities, although an architecture may connect them through several branches or stages.
_Avoid_: equating model computation with the training or evaluation workflow.

**MIL Model Library / MIL 模型库**:
Reusable MIL implementations and their model-specific components, responsible for model computation. It is designed to consume explicit input contracts and remain usable independently of the training and evaluation framework, workflow entry points, and project-specific configuration. A model's mathematical definition and input capabilities are distinct from the workflow that executes it.
_Avoid_: treating project-specific workflow configuration as part of a model's mathematical definition.

**Model Input Compatibility / 模型输入兼容性**:
The relationship between a prepared input and a particular model's requirements for feature structure, instance meaning, and accompanying information. A structure that the library can represent may require a particular model capability; representation alone does not establish that the model can use it.
_Avoid_: treating a representable input structure as proof that every model supports it.

**Input Adapter / Input Mapping / 输入适配器／输入映射**:
The module inside a MIL model that maps incoming feature representations into the representation used by subsequent model computation. It is also called feature projection or instance embedding. Its behavior may preserve original instances or produce derived instances; that correspondence is part of its meaning.
_Avoid_: external feature extraction, file reading, or project-level preprocessing.

**Aggregation Core / 聚合主干**:
The model computation that relates and combines instance representations. It can include attention, spatial relationships, or hierarchical aggregation, rather than only a final pooling operation.
_Avoid_: assuming aggregation is only a final pooling operation.

**Task Head / 任务头**:
The model component that maps a learned representation to task outputs. A head is distinct from the target definition, loss, metrics, and any model-specific auxiliary supervision.
_Avoid_: equating a head with the complete target, loss, and evaluation definition.

**Prediction Task / 预测任务**:
The scientific target and output meaning of a learning problem, such as classification or regression. Choosing an output dimension alone does not fully define a task.
_Avoid_: defining task meaning by output dimension alone.

**Model Baseline / 模型基线**:
A specified model and configuration used as a reference for comparison. The term does not by itself establish equivalence to a paper or an author's implementation.
_Avoid_: assuming a baseline name proves equivalence to a paper or official implementation.

**Scientific Protocol / 科研协议**:
The research choices governing eligible samples, grouping and splits, targets, transformations, model selection, and evaluation. These choices are distinct from the general mechanisms used to execute them.
_Avoid_: equating scientific choices with the mechanisms that execute them.

**Training and Evaluation Framework / 训练与评价框架**:
The reusable execution system that coordinates input access, model use, training, prediction, and evaluation according to declared contracts and a scientific protocol. It uses the model library through explicit interfaces and is designed to be independently usable and portable. Model computation and project-specific scientific preprocessing have separate owners.
_Avoid_: equating execution orchestration with model computation or external preprocessing.

**Prediction / Inference / 预测／推理**:
Applying a selected model to inputs to produce task outputs. Ground-truth labels may be available for subsequent evaluation but are not what produces the prediction.
_Avoid_: treating ground-truth labels as what produces a prediction.

**Evaluation / 评价**:
Measuring predictions against a defined target and evaluation protocol.
_Avoid_: equating prediction generation with measurement against a target.

**Analysis / 分析**:
Deriving comparisons, statistical summaries, interpretations, or visualizations from identified inputs and results. An analysis result has its own meaning and provenance.
_Avoid_: treating a derived interpretation as a replacement for its source predictions.

### Workflows and artifacts

**Workflow Stage / 流程阶段**:
A unit of work with a defined responsibility, declared inputs, owned outputs, and completion conditions. A stage is distinct from the orchestration that connects several stages.
_Avoid_: equating one stage with the orchestration of the entire pipeline.

**Independent Execution / 独立执行**:
Running and checking a stage from its declared, saved, validated inputs without having to rerun preceding stages.
_Avoid_: assuming previous stages must be rerun to execute the current stage.

**Artifact / 产物**:
A saved item with defined meaning, a producer, and identifiable provenance. Data records, configurations, features, model parameters, predictions, and analyses are different kinds of artifacts.
_Avoid_: treating data, configurations, model parameters, and predictions as interchangeable artifacts.

**Producer / Consumer / 生产者／使用者**:
A producer owns the creation of an artifact; a consumer reads it under its contract. A consumer can become the producer of a separate derived artifact.
_Avoid_: equating reading an artifact with owning its creation.

**Derived Artifact / 派生产物**:
A new artifact computed from identified inputs. It has its own producer and lineage rather than replacing the identity or meaning of its sources.
_Avoid_: treating a derived result as a silent replacement for its source.

**Experiment Identity / 实验身份**:
The identity that associates an experiment's selected inputs, protocol, effective configuration, model, and results. A machine-specific filesystem path is a location, not the complete experiment identity.
_Avoid_: equating a machine-specific path with complete experiment identity.

**Run Record / 运行记录**:
The provenance and execution evidence for a particular run, including its inputs, effective configuration, source version, and completion or failure state.
_Avoid_: equating an intended configuration with evidence that a run completed.

**Artifact Contract / 产物约定**:
The declared meaning, required structure, identity correspondence, compatibility, and completeness conditions under which an artifact can be consumed.
_Avoid_: equating file existence with completeness or compatibility.

**Resource-Pool Record / 资源池记录**:
A record describing an available source resource and its relevant attributes. Registration in a resource pool does not imply inclusion in a particular research cohort.
_Avoid_: equating resource registration with cohort inclusion.

**Cohort Selection Manifest / 队列筛选清单**:
The explicit membership selected for a particular research round or protocol from available resources. Selection and resource registration are different facts.
_Avoid_: equating round-specific cohort membership with all available resources.

**Feature Cache / 特征缓存**:
Saved feature representations retained for reuse under their feature-generation identity. Cache membership does not determine cohort inclusion or data splits.
_Avoid_: inferring cohort membership or data splits from cache membership.

**Prediction Artifact / 预测产物**:
Saved model outputs associated with samples, model identity, and the producing run or evaluation context. Training, validation, and external predictions are not interchangeable evidence.
_Avoid_: treating training, validation, and external predictions as interchangeable evidence.

**Analysis Artifact / 分析产物**:
A saved summary, statistical result, comparison, or figure derived from identified data or prediction artifacts. It does not redefine its source predictions.
_Avoid_: redefining source predictions through a derived summary or figure.

### Example Dialogue

**Developer:** An external project flattens an MRI feature grid into a bag before passing it to the library. Is that the Input Adapter?

**Domain explanation:** That is External Preprocessing. Input Adapter / Input Mapping names the module inside the MIL model that maps incoming features into its internal representation.

**Developer:** A bag includes three-dimensional positions. Does that make the model spatially aware?

**Domain explanation:** The positions describe the input. The model uses spatial information only if its computation actually consumes the positions or relationships derived from them. Retaining coordinates in a reader alone does not establish spatial modeling.

**Developer:** The library can represent a bag with three-dimensional positions. Can every MIL model use it?

**Domain explanation:** No. The MIL Input Contract describes the prepared input, while Model Input Compatibility describes whether a particular model can use it. A model's coordinate requirements and supported spatial relationships must be established separately.
