import pandas as pd
import numpy as np
import os
import json
import inspect
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from scipy import stats
from scipy.spatial.distance import pdist

from rdkit import Chem
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, AllChem, rdFreeSASA
from rdkit.Chem.EState import EState
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
from rdkit.Chem.MolStandardize import rdMolStandardize

# --- 全局检测 xTB 库 ---
try:
    from xtb.interface import Calculator, Param

    HAS_XTB = True
except ImportError:
    HAS_XTB = False


# ==============================================================================
# 1. 进阶版构象生成模块 (MMFF 快速粗筛 + GFN2-xTB 量子力学精修)
# ==============================================================================
class ConformationGenerator:
    """3D构象生成器：支持 ETKDGv3 采样、MMFF 过滤及 xTB 量子化学精修"""

    def __init__(self,
                 # --- 采样与搜索参数 ---
                 num_conformers: int = 3000,  # 初始生成的随机构象总数。数值越大，构象空间搜索越完备。
                 max_iterations: int = 800,  # 力场优化（如MMFF94）的最大迭代步数，决定了初步优化的收敛深度。
                 random_seed: int = 42,  # 随机数种子。固定该值可确保每次运行生成的构象结果一致（可重复性）。

                 # --- 冗余剔除与几何筛选 ---
                 rmsd_threshold: float = 0.8,  # RMSD（均方根偏差）阈值。若两个构象差异小于此值，则视为重复并剔除冗余。
                 geometry_check_level: str = 'medium',  # 几何结构检查强度。用于检测是否存在不合理的成键、原子重叠或扭转角。

                 # --- 能量筛选参数（热力学稳定性） ---
                 energy_threshold: float = 10.0,  # 能量绝对截断阈值。剔除能量过高（极不稳定）的异常构象。
                 percentile_cutoff: float = 85.0,  # 百分位筛选。例如 85.0 表示仅保留能量最低（最稳定）的前 85% 构象。
                 use_relative_energy: bool = True,  # 是否启用相对能量筛选（计算 ΔE = E - E_min）。
                 relative_energy_cutoff: float = 8.0,  # 相对能量截断值（单位通常为 kcal/mol）。保留与能量最低构象相比，能级差在此范围内的构象。
                 min_conformers_after_filter: int = 20,  # 经过上述层层筛选后，最少保留的构象数量，确保后续计算有足够的样本量。

                 # --- 物理环境与精修参数 ---
                 temperature: float = 298.15,  # 环境温度（单位：K）。用于计算玻尔兹曼加权分布或构象占据率。
                 use_xtb: bool = True,  # 是否启用 xTB 软件进行半经验量子化学精修。开启后精度远高于普通力场优化。
                 max_xtb_confs: int = 30):  # 允许送入 xTB 进行昂贵计算的最大构象数。由于量子化学计算耗时较长，笔记本电脑建议限制在 20-30 个。

        self.num_conformers = num_conformers
        self.max_iterations = max_iterations
        self.rmsd_threshold = rmsd_threshold
        self.energy_threshold = energy_threshold
        self.random_seed = random_seed
        self.percentile_cutoff = percentile_cutoff
        self.geometry_check_level = geometry_check_level
        self.use_relative_energy = use_relative_energy
        self.relative_energy_cutoff = relative_energy_cutoff
        self.min_conformers_after_filter = min_conformers_after_filter
        self.temperature = temperature
        self.use_xtb = use_xtb and HAS_XTB
        self.max_xtb_confs = max_xtb_confs

    def _refine_with_xtb(self, mol, conf_id):
        """对单个 RDKit 构象执行 GFN2-xTB 单点能计算（兼容 Windows 简化版）"""
        from xtb.interface import Calculator, Param
        try:
            # 提取原子序数和坐标
            atoms = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
            coords = mol.GetConformer(conf_id).GetPositions()

            # 初始化计算器
            calc = Calculator(Param.GFN2xTB, atoms, coords)

            # --- 核心修改点：如果不支持 optimize，则调用 singlepoint ---
            if hasattr(calc, 'optimize'):
                res = calc.optimize(level="normal")
            else:
                # 执行单点能计算：计算当前结构在量子力学水平下的精确能量
                res = calc.singlepoint()

            # 获取能量并转换为 kcal/mol
            energy_kcal = res.get_energy() * 627.509
            return energy_kcal
        except Exception as e:
            self.logger.warning(f"构象 {conf_id} xTB 精修失败: {e}")
            return None

    def generate_conformations(self, mol: Chem.Mol) -> Dict[str, Any]:
        results = {'success': False, 'message': '', 'optimized_mol': None,
                   'conformer_stats': {}, 'boltzmann_weights': []}

        try:
            mol_h = Chem.AddHs(mol)
            self.logger.info(f"第一阶段：生成 {self.num_conformers} 个 ETKDGv3 构象...")

            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.numThreads = 0
            cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=self.num_conformers, params=params)

            if not cids:
                results['message'] = "初始构象生成失败"
                return results

            self.logger.info(f"第二阶段：MMFF94s 快速能量粗筛...")
            opt_res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=self.max_iterations)

            # 整理 MMFF 优化结果并过滤
            opt_results = []
            for cid, (not_conv, energy) in enumerate(opt_res):
                geo_ok = self._check_geometry_vectorized(mol_h, cid) if self.geometry_check_level != 'none' else True
                opt_results.append({'conf_id': cid, 'energy': energy, 'geometry_ok': geo_ok})

            geo_filtered = [r for r in opt_results if r['geometry_ok']]
            if len(geo_filtered) < 2:
                results['message'] = "几何检查后有效构象不足"
                return results

            # 能量窗过滤
            all_e = [r['energy'] for r in geo_filtered]
            e_limit = min(np.percentile(all_e, self.percentile_cutoff), min(all_e) + self.relative_energy_cutoff)
            e_filtered = [r for r in geo_filtered if r['energy'] <= e_limit]

            # Butina 聚类：将 3000 个构象压缩至代表性构象
            clustering_info = self._cluster_conformers(mol_h, e_filtered)
            representative_ids = self._select_representative_conformers(e_filtered, clustering_info)

            if len(representative_ids) > self.max_xtb_confs:
                representative_ids = representative_ids[:self.max_xtb_confs]

            # 第三阶段：量子力学精修 (xTB)
            final_energies = []
            if self.use_xtb:
                self.logger.info(f"第三阶段：对 {len(representative_ids)} 个构象进行 GFN2-xTB 量子精修...")
                for rid in representative_ids:
                    e_xtb = self._refine_with_xtb(mol_h, rid)
                    if e_xtb is None:  # 如果 xTB 失败，回退使用 MMFF 能量
                        e_xtb = next(r['energy'] for r in e_filtered if r['conf_id'] == rid)
                    final_energies.append(e_xtb)
            else:
                final_energies = [next(r['energy'] for r in e_filtered if r['conf_id'] == rid) for rid in
                                  representative_ids]

            # 计算玻尔兹曼权重
            weights = self._calc_weights(final_energies)

            # 构建最终分子（只保留精修过的代表性构象）
            final_mol = Chem.RWMol(mol_h)
            final_mol.RemoveAllConformers()
            for rid in representative_ids:
                final_mol.AddConformer(mol_h.GetConformer(rid), assignId=True)

            results.update({
                'success': True,
                'optimized_mol': final_mol,
                'boltzmann_weights': weights.tolist(),
                'conformer_stats': {
                    'min_energy': min(final_energies),
                    'energy_std': np.std(final_energies),
                    'representative_conformers': len(representative_ids),
                    'level': 'GFN2-xTB' if self.use_xtb else 'MMFF94s'
                }
            })
        except Exception as e:
            results['message'] = str(e)
            self.logger.error(f"处理失败: {e}")

        return results

    def _calc_weights(self, energies):
        e_arr = np.array(energies)
        rel_e = e_arr - np.min(e_arr)
        exp_terms = np.exp(-rel_e / (0.001987 * self.temperature))
        return exp_terms / np.sum(exp_terms)

    def _check_geometry_vectorized(self, mol, conf_id):
        try:
            pos = mol.GetConformer(conf_id).GetPositions()
            if np.any(pdist(pos) < 0.75): return False
            if np.max(np.ptp(pos, axis=0)) > (0.5 + mol.GetNumAtoms() * 0.8): return False
            return True
        except:
            return False

    def _cluster_conformers(self, mol, filtered_results):
        valid_ids = [r['conf_id'] for r in filtered_results]
        dists = []
        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                dists.append(AllChem.GetConformerRMS(mol, valid_ids[i], valid_ids[j], prealigned=True))
        clusters = Butina.ClusterData(dists, len(valid_ids), self.rmsd_threshold, isDistData=True, reordering=True)
        return {'clusters': [[valid_ids[idx] for idx in c] for c in clusters]}

    def _select_representative_conformers(self, e_filtered, clustering_info):
        rep_ids = []
        e_map = {r['conf_id']: r['energy'] for r in e_filtered}
        for cluster in clustering_info['clusters']:
            rep_ids.append(min(cluster, key=lambda cid: e_map[cid]))
        return rep_ids

    @property
    def logger(self):
        if not hasattr(self, '_logger'): self._logger = logging.getLogger('ConfGen')
        return self._logger


# ==============================================================================
# 2. 特征提取模块（结构标准化 + EState原子无关化）
# ==============================================================================
class HDIFeatureExtractor:
    def __init__(self, smiles: str, name: str, **kwargs):
        self.smiles = smiles
        self.name = name
        self.conf_generator = ConformationGenerator(**kwargs)
        self.setup_logging()
        self.logger = logging.getLogger('HDIExtractor')
        self.output_dir = f"{name}_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.features = {}

    def setup_logging(self):
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler(),
                                          logging.FileHandler(f'{self.name}_extraction.log', encoding='utf-8')])

    def _standardize_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """结构标准化：去盐、中和、规范互变异构"""
        try:
            mol = rdMolStandardize.Cleanup(mol)
            mol = rdMolStandardize.FragmentParent(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol)
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
            return mol
        except:
            return mol

    def extract_features(self) -> Dict[str, Any]:
        self.logger.info(f"开始处理: {self.name}")
        raw_mol = Chem.MolFromSmiles(self.smiles)
        if not raw_mol: raise ValueError("SMILES 解析失败")

        mol = self._standardize_molecule(raw_mol)
        conf_res = self.conf_generator.generate_conformations(mol)
        if not conf_res['success']: return {}

        opt_mol = conf_res['optimized_mol']
        weights = conf_res['boltzmann_weights']

        # 加权提取描述符
        raw_features = self._calculate_weighted_descriptors(opt_mol, weights)
        raw_features.update(conf_res['conformer_stats'])
        raw_features.update(self._get_basic_molecule_info(opt_mol))
        raw_features.update(self._calculate_charge_features(mol))
        raw_features.update(self._calculate_surface_features(opt_mol, weights))

        self.features = self._flatten_dict(raw_features)
        self.features.update(self._add_derived_features(self.features, opt_mol))

        self.logger.info(f"特征提取完成，共 {len(self.features)} 个特征。")
        return self.features

    def _get_basic_molecule_info(self, mol):
        return {
            'molecule_name': self.name, 'smiles': self.smiles,
            'formula': rdMolDescriptors.CalcMolFormula(mol),
            'num_atoms': mol.GetNumAtoms(),
            'molecular_weight': Descriptors.MolWt(mol)
        }

    def _calculate_charge_features(self, mol):
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())]
            return {'charge_mean': np.mean(charges), 'charge_std': np.std(charges), 'charge_max': np.max(charges)}
        except:
            return {}

    def _calculate_surface_features(self, mol, weights):
        try:
            radii = rdFreeSASA.classifyAtoms(mol)
            sasa_vals = [rdFreeSASA.CalcSASA(mol, radii, confIdx=i) for i in range(mol.GetNumConformers())]
            return {'SASA_weighted': float(np.average(sasa_vals, weights=weights))}
        except:
            return {}

    def _calculate_weighted_descriptors(self, mol, weights):
        results = {}
        n_conf = mol.GetNumConformers()
        w_array = np.array(weights)
        modules = [('Descriptors', Descriptors), ('rdMD', rdMolDescriptors), ('Lipinski', Lipinski),
                   ('Crippen', Crippen), ('EState', EState)]
        d3d_keys = ['PMI', 'NPR', 'RadiusOfGyration', 'Spherocity', 'PBF', 'WHIM', 'GETAWAY', 'MORSE', 'RDF',
                    'AUTOCORR3D']

        results['Special.BalabanJ'] = BalabanJ(mol)
        results['Special.BertzCT'] = BertzCT(mol)

        for mod_name, module in modules:
            for func_name in dir(module):
                if func_name.startswith('_'): continue
                func = getattr(module, func_name)
                if not callable(func): continue

                try:
                    if func_name == 'EStateIndices':
                        vals = func(mol)
                        results[f"{mod_name}.EState_Max"] = float(np.max(vals))
                        results[f"{mod_name}.EState_Min"] = float(np.min(vals))
                        results[f"{mod_name}.EState_Mean"] = float(np.mean(vals))
                        results[f"{mod_name}.EState_Std"] = float(np.std(vals))
                        continue

                    sig = inspect.signature(func)
                    is_3d = 'confId' in sig.parameters or any(k in func_name for k in d3d_keys)

                    if is_3d and n_conf > 0:
                        conf_vals = []
                        for i in range(n_conf):
                            try:
                                val = func(mol, confId=i) if 'confId' in sig.parameters else func(mol)
                                conf_vals.append(val)
                            except:
                                conf_vals.append(None)
                        results[f"{mod_name}.{func_name}"] = self._weighted_agg(conf_vals, w_array)
                    else:
                        results[f"{mod_name}.{func_name}"] = func(mol)
                except:
                    continue
        return results

    def _weighted_agg(self, values, weights):
        valid_idx = [i for i, v in enumerate(values) if v is not None]
        if not valid_idx: return None
        v_weights = weights[valid_idx] / np.sum(weights[valid_idx])
        v_data = [values[i] for i in valid_idx]
        if isinstance(v_data[0], (list, tuple, np.ndarray)):
            arr = np.array(v_data)
            return np.average(arr, axis=0, weights=v_weights).tolist() if arr.ndim > 1 else float(
                np.average(arr, weights=v_weights))
        return float(np.average(v_data, weights=v_weights))

    def _flatten_dict(self, d):
        flat = {}
        for k, v in d.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                for i, val in enumerate(v): flat[f"{k}_{i + 1}"] = val
            else:
                flat[k] = v
        return flat

    def _add_derived_features(self, feats, mol):
        d = {}
        try:
            mw = feats.get('Descriptors.MolWt', 1)
            rot = Descriptors.NumRotatableBonds(mol)
            d['Flexibility_Index'] = rot / (mw / 100.0)
            d['Shape_Anisotropy'] = feats.get('rdMD.CalcPMI3', 0) / feats.get('rdMD.CalcPMI1', 1)
        except:
            pass
        return d

    def save_results(self):
        df = pd.DataFrame([self.features])
        df.to_csv(os.path.join(self.output_dir, f"{self.name}_full.csv"), index=False)
        excel_path = os.path.join(self.output_dir, f"{self.name}_categorized.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Features', index=False)
            categories = {
                'PhysChem_物理化学': ['Weight', 'MolWt', 'LogP', 'MR', 'TPSA', 'Labute', 'MolVol'],
                'Structure_结构特性': ['Ring', 'Hetero', 'Rotatable', 'Bond', 'Atom', 'Heavy'],
                '3D_Geometry_几何形状': ['PMI', 'NPR', 'RadiusOfGyration', 'Spherocity', 'PBF'],
                'EState_电性状态': ['EState']
            }
            for sheet, keys in categories.items():
                cols = [c for c in df.columns if any(k in c for k in keys)]
                if cols:
                    final = ['molecule_name'] + [c for c in cols if c != 'molecule_name']
                    df[[c for c in final if c in df.columns]].to_excel(writer, sheet_name=sheet[:30], index=False)


class BatchFeatureExtractor:
    def extract_batch(self, molecules: List[Dict[str, str]], output_dir: str = "batch_results", **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        dfs = []
        for i, mol_info in enumerate(molecules):
            name, smi = mol_info.get('name', f'Mol_{i}'), mol_info.get('smiles', '')
            if not smi: continue
            try:
                extractor = HDIFeatureExtractor(smiles=smi, name=name, **kwargs)
                feats = extractor.extract_features()
                if feats:
                    extractor.save_results()
                    dfs.append(pd.DataFrame([feats]))
            except Exception as e:
                print(f"处理 {name} 出错: {e}")

        if dfs:
            aligned_df = pd.concat(dfs, ignore_index=True).fillna(0.0)
            aligned_df.to_csv(os.path.join(output_dir, "aligned_features.csv"), index=False)
            print(f">>> 批量处理完成！对齐特征数: {len(aligned_df.columns)}")


def main():
    print("=" * 60 + "\nPU 分子特征提取旗舰版 V6.1 (xTB 量子力学精修)\n" + "=" * 60)
    mode = input("1. 单分子\n2. 批量输入\n请选择: ").strip()

    if mode == "1":
        smi = input("SMILES: ").strip() or "O=C=NCCCCCCN=C=O"
        name = input("名称: ").strip() or "HDI_Test"

        # 修正：实例化一次，赋值给变量 extractor
        extractor = HDIFeatureExtractor(smiles=smi, name=name)

        # 用同一个实例提取特征
        extractor.extract_features()

        # 用同一个实例保存结果
        extractor.save_results()
    elif mode == "2":
        mols = []
        print("输入 'SMILES,名称'，输入 'done' 结束")
        while True:
            inp = input("> ").strip()
            if inp.lower() == 'done': break
            if ',' in inp:
                s, n = inp.split(',', 1)
                mols.append({'smiles': s.strip(), 'name': n.strip()})
        BatchFeatureExtractor().extract_batch(mols)


if __name__ == "__main__":
    main()