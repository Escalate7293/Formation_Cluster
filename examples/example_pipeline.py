import os
import pandas as pd
from astropy.stats import sigma_clipped_stats, mad_std
import numpy as np

# 从包中引入我们的主体类
from Formation_Cluster_class.batch_imfitter import BatchImfitter, Formation_Cluster

def main():
    # ==========================================
    # 1. 准备测试路径和参数
    # 如果你要测其他源，请在这里替换为正确的 fits 文件绝对路径
    # ==========================================
    # 注意：运行前请确保把这些路径改成真实存在的文件
    test_fits_normal = "/home/esker7293/Cluster_formation/Hotdisk/continuum_detailed/G339.Band6.cycle7.contin.selfcal.image.tt0.fits" # 请修改为真实存在的文件
    test_fits_rbm05 = "/home/esker7293/Cluster_formation/Hotdisk/continuum_detailed/G339.Band6.cycle7.contin.selfcal.robust-0.5.image.tt0.fits" # 请修改为真实存在的文件
    test_csv_path = os.path.join(os.path.dirname(__file__), "g339_b6_c7_venn_sources_v2_less.csv")
    output_directory = os.path.join(os.path.dirname(__file__), "test_output")

    # ==========================================
    # 2. 实例化流水线
    # ==========================================
    print("正在初始化 BatchImfitter 实例...")
    fitter = BatchImfitter(
        output_dir=output_directory,
        distance_pc=2170.0,
        cluster_name="G339_cycle7_test"
    )

    # 检查如果你没有那个 dummy_normal 文件就会报错，可以直接在这边抛出提示
    if not os.path.exists(test_fits_normal):
        print(f"[警告]: {test_fits_normal} 不存在。请打开本脚本，将 test_fits_normal 修改为您真实的 fits 路径后再运行！")
        return

    # ==========================================
    # 3. 运行管线核心逻辑
    # ==========================================
    print("开始运行核心拟合管线 (run_pipeline)...")

    # fc_norm = Formation_Cluster(test_fits_normal)
    # _,_,std_normal = sigma_clipped_stats(fc_norm.img, sigma=3.0, maxiters=None)
    # fc_rbm05 = Formation_Cluster(test_fits_rbm05)
    # _,_,std_rbm05 = sigma_clipped_stats(fc_rbm05.img, sigma=3.0, maxiters=None)
    std_normal = 2e-5 # 加快测试速度，直接用Carta测量的结果，实际使用时建议用 sigma_clipped_stats 重新测量
    std_rbm05 = 3.6e-5 # 加快测试速度，直接用Carta测量的结果，实际使用时建议用 sigma_clipped_stats 重新测量
    
    # 假设你只测试 normal 图，不传入 rbm05 
    final_results = fitter.run_pipeline(
        clustername="G339_cycle7_test",
        source_list_csv=test_csv_path,
        fits_normal=test_fits_normal,
        std_normal=std_normal,
        fits_rbm05=test_fits_rbm05,
        std_rbm05=std_rbm05,
        cutout_size=(100, 100),
        show_plots=True  # 测试环境为了速度，可以先关闭弹窗
    )

    fitter.save_results_to_csv()
    # np.save(os.path.join(output_directory, "final_results.npy"), fitter.results)
    # save_path = os.path.join(output_directory, f'{fitter.cluster_name}_results.pkl')

    # with open(save_path, 'wb') as f:
    #         pickle.dump(fitter.results, f)


if __name__ == "__main__":
    main()
