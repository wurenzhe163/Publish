# DEM-Based Radar Incidence Angle Tracking for Distortion Analysis Without Orbital Data

**By Renzhe Wu (吴仁哲)**  
📧 **Contact**: rswrz@hnas.ac.cn

---

## 📖 About This Research

This project presents a DEM-based approach for geometric distortion detection in SAR imagery that operates without requiring satellite orbital state vector information. The methodology was developed to address challenges in SAR data processing for complex mountainous terrain.

---

## 📚 Citing This Work

If you use this methodology in your research, please cite:

**Wu, R.**, Liu, G., Lv, J., Bao, X., Hong, R., Yang, Z., Wu, S., Xiang, W., & Zhang, R. (2024). *DEM-based radar incidence angle tracking for geometric distortion detection without orbit state information*. **IEEE Transactions on Geoscience and Remote Sensing**, 62, 1-13.  
🎯 **DOI**: [https://doi.org/10.1109/TGRS.2024.3456118](https://doi.org/10.1109/TGRS.2024.3456118)

> **📖 Research Summary**: This study introduces a DEM-based radar incidence angle-tracking method for geometric distortion detection that operates without satellite orbital state vector information. The approach uses ray tracing principles to identify distortion-prone areas in SAR imagery.

---

## 🎯 Research Objectives

This research addresses the challenge of geometric distortion detection in SAR imagery by:
- Developing a method that works without orbital state vector requirements
- Providing an alternative approach for distortion analysis in data-limited scenarios
- Demonstrating the effectiveness of DEM-based radar incidence angle tracking
- Offering a practical solution for mountainous terrain analysis

---

## 🌐 Applications

The methodology has potential applications in:
- **Geological mapping** in mountainous regions
- **Environmental monitoring** where orbital data may be limited
- **SAR data preprocessing** for complex terrain
- **Geometric distortion assessment** for various SAR applications

---

## 📂 Project Structure

```
📁 DEM-Based Radar Incidence Angle Tracking/
├── 📁 GEE_Func/                              # Google Earth Engine functions
│   ├── 📄 GEEMath.py                        # Mathematical operations
│   ├── 📄 GEE_CorreterAndFilters.py         # Correction and filtering
│   ├── 📄 GEE_DataIOTrans.py                # Data I/O and transformation
│   ├── 📄 GEE_Extract_algorithm.py          # Extraction algorithms
│   ├── 📄 GEE_Tools.py                      # Utility tools
│   ├── 📄 S1_distor_dedicated.py            # S1 distortion processing
│   ├── 📄 S2_filter.py                      # S2 filtering functions
│   ├── 📄 download_dem.py                   # DEM download utilities
│   └── 📄 __init__.py                       # Package initialization
├── 📄 SAR_Geometric_Distortion_Analysis.py     # Main analysis script
├── 📄 dem_sampling_methods_comparison_optimized.py  # DEM comparison methods
└── 📖 This documentation
```

---

## 🚀 Methodology Overview

The approach utilizes DEM-based radar incidence angle tracking to identify geometric distortion-prone areas in SAR imagery. By calculating local incidence angles using digital elevation models, the method can detect areas susceptible to layover and shadow effects without requiring satellite orbital information.

---

## 🤝 Collaboration & Contact

I'm interested in research collaboration and discussions about SAR processing methodologies.

**Contact Information:**
- 📧 **Email**: rswrz@hnas.ac.cn
- 🏢 **Institution**: Hunan Academy of Agricultural Sciences
- 🔬 **Research Focus**: SAR remote sensing, geometric distortion analysis

---

## 📖 How to Cite

If you use this methodology in your research, please cite:

```
Wu, R., Liu, G., Lv, J., Bao, X., Hong, R., Yang, Z., Wu, S., Xiang, W., & Zhang, R. (2024). 
DEM-based radar incidence angle tracking for geometric distortion detection without orbit state information. 
IEEE Transactions on Geoscience and Remote Sensing, 62, 1-13. 
https://doi.org/10.1109/TGRS.2024.3456118
```

---

## 🙏 Acknowledgments

This research represents collaborative work with multiple institutions. The methodology development benefited from discussions with colleagues and the support of research institutions involved in this project.

---

*Last Updated: December 2024*