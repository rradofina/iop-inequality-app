# IOP Inequality Analysis Tool - Streamlit App

A streamlined web application for analyzing Inequality of Opportunity (IOP) using Conditional Inference Trees, Random Forests, and Shapley decomposition.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🚀 Quick Start

### Option 1: Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/iop-streamlit-app.git
cd iop-streamlit-app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### Option 2: Deploy on Streamlit Cloud (FREE)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy this repository
5. Share the public URL with anyone!

## 📊 Features

- **📁 Easy Data Upload**: Drag and drop CSV/Excel files
- **🎯 C-Tree Analysis**: Partition population into types
- **🌲 C-Forest Analysis**: Ensemble method for robust estimation
- **🎲 Shapley Decomposition**: Decompose inequality by circumstances
- **📈 Interactive Visualizations**: Plotly charts and tables
- **💾 Export Results**: Download as JSON or CSV
- **⚡ Fast Performance**: Optimized algorithms
- **🎨 Beautiful UI**: Clean, professional interface

## 📋 Data Requirements

Your data file should include:

| Column | Description | Required |
|--------|-------------|----------|
| `income` | Income values (positive) | ✅ Yes |
| `weights` | Sample weights | ⚡ Optional |
| `Sex` | Gender (0/1 or 1/2) | 📊 Circumstance |
| `Father_Edu` | Father's education level | 📊 Circumstance |
| `Mother_Edu` | Mother's education level | 📊 Circumstance |
| `Birth_Area` | Birth area/region | 📊 Circumstance |
| `Religion` | Religious affiliation | 📊 Circumstance |

## 🖥️ Usage

1. **Upload Data**: Use the sidebar to upload your CSV/Excel file
2. **Configure**: Select circumstance variables and analysis options
3. **Run Analysis**: Click the "Run Analysis" button
4. **View Results**: Explore interactive charts and tables
5. **Export**: Download results in JSON or CSV format

## 🔧 Configuration

### Streamlit Config (Optional)

Create `.streamlit/config.toml` for custom theming:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 🐳 Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t iop-app .
docker run -p 8501:8501 iop-app
```

## 📚 Methodology

This tool implements ex-ante IOP analysis based on:

- **Roemer (1998)**: Equality of Opportunity framework
- **Ferreira & Gignoux (2011)**: Ex-ante measurement approach
- **ADB Workshop**: Practical implementation guidelines

### Key Concepts:

- **Circumstances**: Factors beyond individual control (e.g., parental education, birthplace)
- **Types**: Groups with identical circumstances
- **IOP**: Share of inequality attributable to circumstances
- **Shapley Values**: Marginal contribution of each circumstance

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - feel free to use this tool for research and education.

## 🙏 Acknowledgments

Based on the IOP methodology from the ADB Workshop on Inequality of Opportunity.

## 📧 Contact

For questions or support:
- Open an issue on GitHub
- Email: your.email@example.com

## 🎯 Roadmap

- [ ] Add more inequality measures (Theil T, Atkinson)
- [ ] Support for panel data
- [ ] Bootstrap confidence intervals
- [ ] Export report generation (PDF)
- [ ] Multi-language support

---

**Made with ❤️ using Streamlit**