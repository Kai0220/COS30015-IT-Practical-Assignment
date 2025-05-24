# Tracing and Classifying Spam Emails  
**A Machine Learning and Threat Intelligence Approach**

This project presents the full implementation and evaluation of a dual-layered system designed to detect, trace, and analyse spam emails. It integrates machine learning-based classification (Logistic Regression) with threat intelligence techniques such as IP geolocation, WHOIS domain lookups, and spam-type categorisation.

The system is built to not only differentiate spam from legitimate emails, but also to uncover the origin, distribution patterns, and behavioural characteristics of spam campaigns. It identifies weaknesses in conventional detection systems by analysing false negatives and providing insight into their causes. The project aligns with the objectives of attacker and malware analysis, under the COS30015 Practical Research Project (S1_2025), with a strong focus on practical cybersecurity application.

---

### 👤 Prepared by  
**Jesse Ting Wen Kai**  
*Student ID: 102769808*

---

### 📁 Dataset  
- [Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)

---

### 🛠️ Tools & APIs Used  
- [python-whois](https://pypi.org/project/python-whois/)  
- [IP-API](https://ip-api.com/)  
- [VirusTotal](https://www.virustotal.com/gui/home/upload)  
- [Scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
