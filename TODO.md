# CliniQ AI v2.0 Deployment on Render

## ✅ Completed Tasks
- [x] Updated config.py to support DATABASE_URL environment variable
- [x] Added psycopg2-binary to requirements.txt for PostgreSQL support
- [x] Verified render.yaml configuration
- [x] Verified Procfile configuration

## 🚀 Deployment Steps

### 1. Prepare GitHub Repository
- [ ] Commit all changes to your GitHub repository
- [ ] Push the latest code to GitHub
- [ ] Ensure all files are included (especially model/ directory)

### 2. Set Up Render Account
- [ ] Go to https://render.com and sign up/login
- [ ] Connect your GitHub account to Render

### 3. Create PostgreSQL Database
- [ ] In Render dashboard, click "New" → "PostgreSQL"
- [ ] Choose a name (e.g., "cliniq-ai-db")
- [ ] Select free tier or paid plan as needed
- [ ] Note down the "Internal Database URL" and "External Database URL"

### 4. Deploy Web Service
- [ ] In Render dashboard, click "New" → "Web Service"
- [ ] Connect your GitHub repository
- [ ] Select the branch to deploy from (usually main/master)
- [ ] Configure the service:
  - **Name**: cliniq-ai (or your preferred name)
  - **Runtime**: Python 3
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `gunicorn app:app`
- [ ] Set environment variables:
  - `DATABASE_URL`: Your PostgreSQL database URL (use Internal URL)
  - `FLASK_ENV`: production
  - `SECRET_KEY`: Generate a secure random key
- [ ] Click "Create Web Service"

### 5. Monitor Deployment
- [ ] Wait for the build to complete (may take 10-15 minutes)
- [ ] Check the deployment logs for any errors
- [ ] If build fails, check logs and fix issues

### 6. Test Deployment
- [ ] Once deployed, visit the provided URL
- [ ] Test user registration and login
- [ ] Test prediction features with sample images
- [ ] Verify database connectivity

### 7. Post-Deployment Configuration
- [ ] Set up custom domain (optional)
- [ ] Configure monitoring and alerts
- [ ] Set up automatic deployments from GitHub

## 🔧 Troubleshooting

### Common Issues:
- **Build fails due to missing dependencies**: Check requirements.txt
- **Database connection errors**: Verify DATABASE_URL is correct
- **Model loading errors**: Ensure model/ directory is included in deployment
- **Port binding errors**: Ensure app listens on 0.0.0.0:$PORT

### Environment Variables Reference:
```
DATABASE_URL=postgresql://user:password@host:port/database
FLASK_ENV=production
SECRET_KEY=your-secure-random-key-here
```

## 📋 Notes
- Free Render tier has limitations (750 hours/month, sleeps after 15min inactivity)
- ML models may take time to load on first request
- Consider upgrading to paid tier for production use
- Monitor usage and costs regularly
