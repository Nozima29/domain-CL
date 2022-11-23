import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
from utils import get_data_loader, checkattr
from data.manipulate import SubDataset, MemorySetDataset
from models.cl.continual_learner import ContinualLearner


def train_cl(model, train_datasets, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), **kwargs):
    model.train()
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st context)
    ReplayStoredData = ReplayGeneratedData = ReplayCurrentData = False
    previous_model = None

    # Register starting parameter values (needed for SI)
    if isinstance(model, ContinualLearner) and model.importance_weighting == 'si':
        model.register_starting_param_values()

    # Are there different active classes per context (or just potentially a different mask per context)?
    per_context = False
    per_context_singlehead = False

    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):

        # If using the "joint" baseline, skip to last context, as model is only be trained once on data of all contexts
        if baseline == 'joint':
            if context < len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # Add memory buffer (if available) to current dataset (if requested)
        if checkattr(model, 'add_buffer') and context > 1:
            if model.scenario == "domain" or per_context_singlehead:
                target_transform = (
                    lambda y, x=model.classes_per_context: y % x)
            else:
                target_transform = None
            memory_dataset = MemorySetDataset(
                model.memory_sets, target_transform=target_transform)
            training_dataset = ConcatDataset([train_dataset, memory_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update (needed for SI)
        if isinstance(model, ContinualLearner) and model.importance_weighting == 'si':
            W, p_old = model.prepare_importance_estimates_dicts()

        active_classes = None

        # Reset state of optimizer(s) for every context (if requested)
        if (not model.label == "SeparateClassifiers") and model.optim_type == "adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type == "adam_reset":
            generator.optimizer = optim.Adam(
                model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if per_context:
            up_to_context = context if baseline == "cummulative" else context-1
            iters_left_previous = [1]*up_to_context
            data_loader_previous = [None]*up_to_context

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        for batch_index in range(1, iters_to_use+1):
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(get_data_loader(
                    training_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)
            if ReplayStoredData:
                if per_context:
                    up_to_context = context if baseline == "cummulative" else context-1
                    batch_size_replay = int(
                        np.ceil(batch_size/up_to_context)) if (up_to_context > 1) else batch_size
                    # -if different active classes per context (e.g., Task-IL), need separate replay for each context
                    for context_id in range(up_to_context):
                        batch_size_to_use = min(batch_size_replay, len(
                            previous_datasets[context_id]))
                        iters_left_previous[context_id] -= 1
                        if iters_left_previous[context_id] == 0:
                            data_loader_previous[context_id] = iter(get_data_loader(
                                previous_datasets[context_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[context_id] = len(
                                data_loader_previous[context_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous == 0:
                        batch_size_to_use = min(batch_size, len(
                            ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(get_data_loader(ConcatDataset(previous_datasets),
                                                                    batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)

            x, y = next(data_loader)
            y = y.classes_per_context * \
                (context-1) if per_context and not per_context_singlehead else y
            x, y = x.to(device), y.to(device)

            binary_distillation = hasattr(
                model, "binaryCE") and model.binaryCE and model.binaryCE_distill
            if binary_distillation and model.scenario in ("class", "all") and (previous_model is not None):
                with torch.no_grad():
                    scores = previous_model.classify(
                        x, no_prototypes=True
                    )[:, :(model.classes_per_context * (context - 1))]
            else:
                scores = None

            if not ReplayStoredData and not ReplayGeneratedData and not ReplayCurrentData:
                x_ = y_ = scores_ = context_used = None

            if ReplayStoredData:
                scores_ = context_used = None
                if not per_context:
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (
                        model.replay_targets == "hard") else None
                    if (model.replay_targets == "soft"):
                        with torch.no_grad():
                            scores_ = previous_model.classify(
                                x_, no_prototypes=True)
                        if model.scenario == "class" and model.neg_samples == "all-so-far":
                            scores_ = scores_[
                                :, :(model.classes_per_context*(context-1))]
                else:
                    x_ = list()
                    y_ = list()
                    up_to_context = context if baseline == "cummulative" else context-1
                    for context_id in range(up_to_context):
                        x_temp, y_temp = next(data_loader_previous[context_id])
                        x_.append(x_temp.to(device))
                        if model.replay_targets == "hard":
                            if not per_context_singlehead:
                                y_temp = y_temp - \
                                    (model.classes_per_context*context_id)
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    if (model.replay_targets == "soft") and (previous_model is not None):
                        scores_ = list()
                        for context_id in range(up_to_context):
                            with torch.no_grad():
                                scores_temp = previous_model.classify(
                                    x_[context_id], no_prototypes=True)
                            if active_classes is not None:
                                scores_temp = scores_temp[:,
                                                          active_classes[context_id]]
                            scores_.append(scores_temp)

            if ReplayCurrentData:
                x_ = x  # --> use current context inputs
                context_used = None

            if ReplayGeneratedData:
                conditional_gen = True if previous_generator.label == 'CondVAE' and \
                    ((previous_generator.per_class and previous_generator.prior == "GMM")
                     or checkattr(previous_generator, 'dg_gates')) else False
                if conditional_gen and per_context:
                    # -if a cond generator is used with different active classes per context, generate data per context
                    x_ = list()
                    context_used = list()
                    for context_id in range(context-1):
                        allowed_domains = list(range(context - 1))
                        allowed_classes = list(
                            range(model.classes_per_context*context_id,
                                  model.classes_per_context*(context_id+1))
                        )
                        batch_size_to_use = int(
                            np.ceil(batch_size / (context-1)))
                        x_temp_ = previous_generator.sample(batch_size_to_use, allowed_domains=allowed_domains,
                                                            allowed_classes=allowed_classes, only_x=False)
                        x_.append(x_temp_[0])
                        context_used.append(x_temp_[2])
                else:
                    allowed_classes = None
                    allowed_domains = list(range(context-1))
                    x_temp_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes,
                                                        allowed_domains=allowed_domains, only_x=False)
                    x_ = x_temp_[0] if type(x_temp_) == tuple else x_temp_
                    context_used = x_temp_[2] if type(
                        x_temp_) == tuple else None

            if ReplayGeneratedData or ReplayCurrentData:
                if not per_context:
                    with torch.no_grad():
                        scores_ = previous_model.classify(
                            x_, no_prototypes=True)
                    if model.scenario == "class" and model.neg_samples == "all-so-far":
                        scores_ = scores_[
                            :, :(model.classes_per_context * (context - 1))]
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    scores_ = list()
                    y_ = list()
                    if previous_model.mask_dict is None and not type(x_) == list:
                        with torch.no_grad():
                            all_scores_ = previous_model.classify(
                                x_, no_prototypes=True)
                    for context_id in range(context-1):
                        if previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(context=context_id+1)
                        if previous_model.mask_dict is not None or type(x_) == list:
                            with torch.no_grad():
                                all_scores_ = previous_model.classify(x_[context_id] if type(x_) == list else x_,
                                                                      no_prototypes=True)
                        temp_scores_ = all_scores_
                        if active_classes is not None:
                            temp_scores_ = temp_scores_[
                                :, active_classes[context_id]]
                        scores_.append(temp_scores_)
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        y_.append(temp_y_)

                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None

            if batch_index <= iters:
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_, rnt=1./context,
                                                contexts_=context_used, active_classes=active_classes, context=context)

                if isinstance(model, ContinualLearner) and model.importance_weighting == 'si':
                    model.update_importance_estimates(W, p_old)

                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index,
                                loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, context=context)

            if generator is not None and batch_index <= gen_iters:
                loss_dict = generator.train_a_batch(x, x_=x_, rnt=1./context)
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index,
                                loss_dict, context=context)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, context=context)

        progress.close()
        if generator is not None:
            progress_gen.close()

        # Parameter regularization: update and compute the parameter importance estimates
        if context < len(train_datasets) and isinstance(model, ContinualLearner):
            allowed_classes = active_classes[-1] if (
                per_context and not per_context_singlehead) else active_classes
            if model.mask_dict is not None:
                model.apply_XdGmask(context=context)
            if model.importance_weighting == 'fisher' and (model.weight_penalty or model.precondition):
                if model.fisher_kfac:
                    model.estimate_kfac_fisher(
                        training_dataset, allowed_classes=allowed_classes)
                else:
                    model.estimate_fisher(
                        training_dataset, allowed_classes=allowed_classes)
            if model.importance_weighting == 'owm' and (model.weight_penalty or model.precondition):
                model.estimate_owm_fisher(
                    training_dataset, allowed_classes=allowed_classes)
            if model.importance_weighting == 'si' and (model.weight_penalty or model.precondition):
                model.update_omega(W, model.epsilon)

        # MEMORY BUFFER: update the memory buffer
        if checkattr(model, 'use_memory_buffer'):
            samples_per_class = model.budget_per_class if (not model.use_full_capacity) else int(
                np.floor((model.budget_per_class*len(train_datasets))/context)
            )
            # reduce examplar-sets (only needed when '--use-full-capacity' is selected)
            model.reduce_memory_sets(samples_per_class)
            new_classes = list(range(model.classes_per_context)) if (
                model.scenario == "domain" or per_context_singlehead
            ) else list(range(model.classes_per_context*(context-1), model.classes_per_context*context))
            for class_id in new_classes:
                class_dataset = SubDataset(
                    original_dataset=train_dataset, sub_labels=[class_id])
                allowed_classes = active_classes[-1] if per_context and not per_context_singlehead else active_classes
                model.construct_memory_set(
                    dataset=class_dataset, n=samples_per_class, label_set=allowed_classes)
            model.compute_means = True
        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)

        # REPLAY: update source for replay
        if context < len(train_datasets) and hasattr(model, 'replay_mode'):
            previous_model = copy.deepcopy(model).eval()
            if model.replay_mode == 'generative':
                ReplayGeneratedData = True
                previous_generator = copy.deepcopy(generator).eval(
                ) if generator is not None else previous_model
            elif model.replay_mode == 'current':
                ReplayCurrentData = True
            elif model.replay_mode in ('buffer', 'all'):
                ReplayStoredData = True
                if model.replay_mode == "all":
                    previous_datasets = train_datasets[:context]
                else:
                    if per_context:
                        previous_datasets = []
                        for context_id in range(context):
                            previous_datasets.append(MemorySetDataset(
                                model.memory_sets[
                                    (model.classes_per_context * context_id):(model.classes_per_context*(context_id+1))
                                ],
                                target_transform=(lambda y, x=model.classes_per_context * context_id: y + x) if (
                                    not per_context_singlehead
                                ) else (lambda y, x=model.classes_per_context: y % x)
                            ))
                    else:
                        target_transform = None if not model.scenario == "domain" else (
                            lambda y, x=model.classes_per_context: y % x
                        )
                        previous_datasets = [MemorySetDataset(
                            model.memory_sets, target_transform=target_transform)]


def train_fromp(model, train_datasets, iters=2000, batch_size=32,
                loss_cbs=list(), eval_cbs=list(), context_cbs=list(), **kwargs):
    model.train()
    cuda = model._is_on_cuda()
    device = model._device()
    per_context = False
    per_context_singlehead = False
    for context, train_dataset in enumerate(train_datasets, 1):
        active_classes = None
        label_sets = active_classes if (per_context and not per_context_singlehead) else [
            active_classes]*context

        if context > 1:
            model.optimizer.init_context(context-1, reset=(model.optim_type == "adam_reset"),
                                         classes_per_context=model.classes_per_context, label_sets=label_sets)

        iters_left = 1
        progress = tqdm.tqdm(range(1, iters+1))
        for batch_index in range(1, iters+1):
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(get_data_loader(
                    train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            x, y = next(data_loader)
            y = y - model.classes_per_context * \
                (context - 1) if (per_context and not per_context_singlehead) else y

            x, y = x.to(device), y.to(device)
            if batch_index <= iters:
                loss_dict = model.optimizer.step(
                    x, y, label_sets, context-1, model.classes_per_context)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index,
                                loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)

        progress.close()

        if checkattr(model, 'use_memory_buffer'):
            samples_per_class = model.budget_per_class if (not model.use_full_capacity) else int(
                np.floor((model.budget_per_class*len(train_datasets))/context)
            )
            model.reduce_memory_sets(samples_per_class)
            new_classes = list(range(model.classes_per_context)) if (
                model.scenario == "domain" or per_context_singlehead
            ) else list(range(model.classes_per_context*(context-1), model.classes_per_context*context))
            for class_id in new_classes:
                class_dataset = SubDataset(
                    original_dataset=train_dataset, sub_labels=[class_id])
                allowed_classes = active_classes[-1] if per_context and not per_context_singlehead else active_classes
                model.construct_memory_set(
                    dataset=class_dataset, n=samples_per_class, label_set=allowed_classes)
            model.compute_means = True

        if context < len(train_datasets):
            memorable_loader = DataLoader(
                dataset=train_dataset, batch_size=6, shuffle=False, num_workers=3)
            model.optimizer.update_fisher(
                memorable_loader,
                label_set=active_classes[context-1] if (
                    per_context and not per_context_singlehead) else active_classes
            )

        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)


def train_gen_classifier(model, train_datasets, iters=2000, epochs=None, batch_size=32,
                         loss_cbs=list(), sample_cbs=list(), eval_cbs=list(), context_cbs=list(), **kwargs):

    device = model._device()
    cuda = model._is_on_cuda()

    classes_in_current_context = 0
    context = 1
    for class_id, train_dataset in enumerate(train_datasets):
        iters_left = 1

        if epochs is not None:
            data_loader = iter(get_data_loader(
                train_dataset, batch_size, cuda=cuda, drop_last=False))
            iters = len(data_loader)*epochs

        progress = tqdm.tqdm(range(1, iters+1))
        for batch_index in range(1, iters+1):
            iters_left -= 1
            if iters_left == 0:
                data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda,
                                                   drop_last=True if epochs is None else False))
                iters_left = len(data_loader)
            x, y = next(data_loader)
            x, y = x.to(device), y.to(device)
            model_to_be_trained = getattr(model, "vae{}".format(class_id))

            loss_dict = model_to_be_trained.train_a_batch(x)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index,
                            loss_dict, class_id=class_id)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index +
                            classes_in_current_context*iters, context=context)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(model_to_be_trained,
                              batch_index, class_id=class_id)

        progress.close()
        classes_in_current_context += 1
        if classes_in_current_context == model.classes_per_context:
            for context_cb in context_cbs:
                if context_cb is not None:
                    context_cb(model, iters, context=context)
            classes_in_current_context = 0
            context += 1
