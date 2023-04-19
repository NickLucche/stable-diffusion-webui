import modules.scripts
from modules import sd_samplers
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


import torch.multiprocessing as mp
from typing import List

def foo(in_q:mp.Queue, out_q: mp.Queue):
    import modules.shared as shared
    import modules.sd_models
    import sys
    from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
    import modules.codeformer_model as codeformer
    import modules.face_restoration
    import modules.gfpgan_model as gfpgan
    import modules.img2img

    import modules.lowvram
    import modules.scripts
    import modules.sd_hijack
    import modules.sd_models
    import modules.sd_vae
    import modules.txt2img
    import modules.script_callbacks
    import modules.textual_inversion.textual_inversion
    import modules.progress

    import modules.ui
    from modules import modelloader
    from modules.shared import cmd_opts
    import modules.hypernetworks.hypernetwork
    # NOTE either you load here or you have main proc load weights and then you copy to cuda:<id>..
    extensions.list_extensions()
    localization.list_localizations(cmd_opts.localizations_dir)
    # startup_timer.record("list extensions")

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    # startup_timer.record("list SD models")

    codeformer.setup_model(cmd_opts.codeformer_models_path)
    # startup_timer.record("setup codeformer")

    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    # startup_timer.record("setup gfpgan")

    modelloader.list_builtin_upscalers()
    # startup_timer.record("list builtin upscalers")

    modules.scripts.load_scripts()
    # startup_timer.record("load scripts")

    modelloader.load_upscalers()
    # startup_timer.record("load upscalers")

    modules.sd_vae.refresh_vae_list()
    # startup_timer.record("refresh VAE")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    # startup_timer.record("refresh textual inversion templates")
    try:
        print("Loading model from proc..")
        modules.sd_models.load_model()
    except Exception as e:
        # errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        return
    
    prompt = "a bear fishing for salmon in a running river"
    while True:
        args = in_q.get(block=True)
        # quit signal
        if args is None:
            break
        print('proc shared model', shared.sd_model.device)
        pipe = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            # outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
            # outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=prompt,
            
            # styles=prompt_styles,
            # negative_prompt=negative_prompt,
            # seed=seed,
            # subseed=subseed,
            # subseed_strength=subseed_strength,
            # seed_resize_from_h=seed_resize_from_h,
            # seed_resize_from_w=seed_resize_from_w,
            # seed_enable_extras=seed_enable_extras,
            # sampler_name=sd_samplers.samplers[sampler_index].name,
            # batch_size=batch_size,
            # n_iter=n_iter,
            # steps=steps,
            # cfg_scale=cfg_scale,
            # width=width,
            # height=height,
            # restore_faces=restore_faces,
            # tiling=tiling,
            # enable_hr=enable_hr,
            # denoising_strength=denoising_strength if enable_hr else None,
            # hr_scale=hr_scale,
            # hr_upscaler=hr_upscaler,
            # hr_second_pass_steps=hr_second_pass_steps,
            # hr_resize_x=hr_resize_x,
            # hr_resize_y=hr_resize_y,
            # override_settings=override_settings,
        )

        pipe.scripts = modules.scripts.scripts_txt2img
        pipe.script_args = args
        pipe.do_not_save_samples = True
        # if cmd_opts.enable_console_prompts:
            # print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

        processed = modules.scripts.scripts_txt2img.run(pipe, *args)

        if processed is None:
            processed = process_images(pipe)

        pipe.close()
        out_q.put(processed)



_procs: List[mp.Process] = []
q, outq = None, None
n_processes = 1
def quit_processes():
    if q is None or not len(_procs):
        return
    print("Quitting processes..")
    for _ in range(n_processes):
        q.put(None)
    for proc in _procs:
        proc.join()
    print("Done!")

def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, override_settings_texts, *args):
    # print('main shared model', shared.sd_model) TODO this must be none
    
    # create models in their own process and move them to correspoding device
    if not len(_procs):
        q = mp.Queue()
        outq = mp.Queue()

        for i in range(n_processes):
            p = mp.Process(
                target=foo,
                args=(q, outq),
                daemon=False,
            )
            p.start()
            _procs.append(p)

    print(args)
    override_settings = create_override_settings_dict(override_settings_texts)

    print("Pushing to processes")
    q.put(args)
    processed: Processed = outq.get(block=True, timeout=None)

    shared.total_tqdm.clear()
    generation_info_js = processed.js()
    # if opts.samples_log_stdout:
    #     print(generation_info_js)

    # if opts.do_not_show_images:
        # processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)
