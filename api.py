from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# Trang chủ
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>PCA và Standardize API</title>
        </head>
        <body>
            <h2>Tải lên dữ liệu CSV để phân tích PCA</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv" required><br><br>
                <label for="column_index">Nhập số thứ tự của cột đầu tiên để phân tích (bắt đầu từ 0):</label>
                <input type="number" name="column_index" min="0" required><br><br>
                <input type="submit" value="Tải lên và phân tích">
            </form>
        </body>
    </html>
    """

# Nhận file và thực hiện phân tích PCA
@app.post("/upload")
async def upload_file(file: UploadFile, column_index: int = Form(...)):
    # Đọc file CSV
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    # Lấy dữ liệu từ cột chỉ định
    data = df.iloc[:, column_index].values.reshape(-1, 1)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    
    # Phân tích PCA
    pca = PCA()
    pca.fit(data_standardized)
    
    # Vẽ scree plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Số thứ tự thành phần chính')
    plt.ylabel('Tỷ lệ phương sai giải thích')
    
    # Lưu hình ảnh scree plot vào buffer
    img_scree = io.BytesIO()
    plt.savefig(img_scree, format='png')
    img_scree.seek(0)
    scree_base64 = base64.b64encode(img_scree.getvalue()).decode('utf-8')
    
    # Vẽ score plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pca.components_[0], pca.components_[1])
    plt.title('Score Plot')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Lưu hình ảnh score plot vào buffer
    img_score = io.BytesIO()
    plt.savefig(img_score, format='png')
    img_score.seek(0)
    score_base64 = base64.b64encode(img_score.getvalue()).decode('utf-8')
    
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>Kết quả PCA</title>
        </head>
        <body>
            <h2>Scree Plot và Score Plot</h2>
            <h3>Scree Plot:</h3>
            <img src="data:image/png;base64,{scree_base64}" alt="Scree Plot" /><br><br>
            <h3>Score Plot:</h3>
            <img src="data:image/png;base64,{score_base64}" alt="Score Plot" />
        </body>
    </html>
    """)
