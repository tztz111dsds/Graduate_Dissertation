# This script processes Airbnb host descriptions by extracting key themes and performing sentiment analysis.
#
# Step 1: The script begins by loading preprocessed host descriptions and a theme keyword dictionary.
#
# Step 2: Using the DistilBERT model, it generates sentence embeddings for predefined themes and for each host description.
#
# Step 3: The script performs an initial theme matching based on keywords found in the descriptions. These keywords are
#         matched against the theme keyword dictionary to generate a list of candidate themes for each description.
#
# Step 4: The list of candidate themes is refined by computing cosine similarity between the review embeddings
#         and the embeddings of each candidate theme. The theme with the highest similarity is assigned to the description.
#
# Step 5: After assigning the final theme, sentiment analysis is performed using the VADER sentiment analyzer.
#         The sentiment scores are scaled from -5 (most negative) to 5 (most positive).
#
# Step 6: Finally, the script calculates the average sentiment score and the number of reviews for each theme.
#         These results are aggregated and saved to a CSV file for further analysis or reporting.
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm  # 进度条
# 加载评论数据
df_reviews = pd.read_csv(r'D:\py\pythonProject\final\preprocessed_hosts.csv')  # 用户评论 CSV 文件
df_dict = pd.read_csv(r'D:\py\pythonProject\final\expanded_theme_keywords.csv')  # 主题词典 CSV 文件

# 初始化 DistilBERT 模型
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 初始化情感分析器
analyzer = SentimentIntensityAnalyzer()

# Descriptive Sentence for Topic
theme_sentences = {
    'Living Space & Comfort': [
        "The living area feels spacious and welcoming, perfect for relaxing.",
        "This cozy space offers a warm and comfortable environment for guests.",
        "The room layout is designed for maximum comfort and ease.",
        "A relaxing ambiance permeates the entire living room.",
        "The furnishings are soft and cozy, making the space feel like home.",
        "The apartment is designed for ultimate relaxation and comfort.",
        "Guests can unwind in the peaceful and quiet living area.",
        "This space is both cozy and functional, ideal for daily living.",
        "The seating arrangement promotes comfort and socializing.",
        "The room is spacious enough for both relaxation and entertainment.",
        "Every piece of furniture is selected for comfort and style.",
        "The soft lighting enhances the cozy atmosphere of the space.",
        "The open layout allows for easy movement and a relaxed feel.",
        "This living space is perfect for unwinding after a long day.",
        "The room is arranged to maximize comfort and tranquility.",
        "The relaxing atmosphere makes this space feel like a peaceful retreat.",
        "The area is thoughtfully designed to promote relaxation.",
        "Cozy corners throughout the room invite guests to sit and relax.",
        "The warm tones and soft textures create a serene living environment.",
        "The layout ensures both privacy and comfort for all guests."
    ],

    'Quality & Design': [
        "This home is designed with a modern, luxurious feel in mind.",
        "High-quality materials are used throughout the property.",
        "The aesthetic is both elegant and sophisticated.",
        "The attention to detail in the design is impeccable.",
        "Every room is filled with designer touches and luxurious finishes.",
        "The property exudes a sense of elegance and refinement.",
        "This space is designed to impress with its sleek and modern look.",
        "The craftsmanship is evident in every corner of the property.",
        "The design is both functional and beautifully artistic.",
        "Premium materials and fixtures elevate the entire space.",
        "The décor is tasteful, with a perfect blend of modern and traditional elements.",
        "This home offers a stylish and chic living experience.",
        "Every piece of furniture and décor is chosen with precision.",
        "The modern design is complemented by luxurious details.",
        "The interior is beautifully crafted with high-end finishes.",
        "The quality of the design is apparent in every detail.",
        "The home is filled with custom, high-end furnishings.",
        "This space is both artistic and luxurious in design.",
        "The design perfectly balances style and functionality.",
        "The sophisticated décor creates a timeless, luxurious feel."
    ],

    'Family-Focused': [
        "This home is perfect for families, with plenty of space for everyone.",
        "The property includes child-safe features to ensure peace of mind for parents.",
        "Family-friendly amenities like a crib and high chair are available.",
        "The space is designed to be comfortable and safe for children.",
        "There are plenty of kid-friendly activities and toys provided.",
        "The home is spacious enough for families to enjoy together.",
        "This is an ideal space for families with young children.",
        "Families can enjoy the large, open spaces perfect for bonding.",
        "The home is equipped with everything a family might need.",
        "The layout is family-friendly, ensuring easy supervision of children.",
        "There are designated play areas for kids to enjoy.",
        "The space is comfortable and welcoming for families of all sizes.",
        "Safety features throughout the home make it perfect for young children.",
        "Family-friendly furniture and amenities ensure a comfortable stay.",
        "This property is ideal for families looking for a safe, comfortable home.",
        "There is plenty of room for families to spread out and relax.",
        "The property is equipped with family essentials, from toys to cribs.",
        "Families will feel right at home in this comfortable, child-safe environment.",
        "Parents can relax knowing the home is fully childproofed.",
        "There’s something for every family member to enjoy in this spacious home."
    ],
    'User Experience': [
        "Guests rave about the seamless check-in process and quick communication.",
        "The host goes above and beyond to ensure every guest's satisfaction.",
        "The communication with the host was timely and professional.",
        "The check-in process was smooth and straightforward.",
        "Every guest has reported a wonderful experience staying here.",
        "The host is responsive and attentive to guest needs.",
        "Guests appreciate the clear instructions provided by the host.",
        "The overall experience exceeded guest expectations.",
        "The host ensured that the stay was comfortable and stress-free.",
        "The check-out process was easy and hassle-free.",
        "Guests appreciate the host’s availability and quick responses.",
        "The host’s attention to detail makes every stay memorable.",
        "Guests always feel welcome and well taken care of.",
        "The property is exactly as described, which makes for a smooth experience.",
        "Guests feel at home thanks to the host's exceptional service.",
        "The host was accommodating, ensuring every need was met.",
        "The guest experience is top-notch, from booking to check-out.",
        "Guests frequently compliment the host’s hospitality.",
        "Everything about the stay went smoothly, thanks to the host.",
        "The overall guest experience is consistently rated five stars."
    ],

    'Trust & Assurance': [
        "Guests feel secure knowing the property is accurately described.",
        "The host is trustworthy and delivers exactly as promised.",
        "Guests appreciate the host's reliability and attention to detail.",
        "The property is exactly as shown in the pictures, which builds trust.",
        "Guests feel safe and assured with the host's responsiveness.",
        "The host provides accurate information, building guest confidence.",
        "Trust is built through clear communication and transparent processes.",
        "The property’s safety measures ensure guest peace of mind.",
        "Guests can rely on the host to provide everything they need.",
        "The host's reputation for honesty is a huge plus.",
        "Guests know they can trust the host for an accurate experience.",
        "The reliable service makes guests feel confident in their booking.",
        "Every interaction with the host builds trust and assurance.",
        "Safety features in the home contribute to guest satisfaction.",
        "The host's reliability makes guests feel secure during their stay.",
        "Guests appreciate the accurate, honest descriptions of the property.",
        "The host consistently delivers a trustworthy experience.",
        "Guests feel confident booking, knowing the host is reputable.",
        "Every detail is exactly as promised, building guest trust.",
        "The host ensures every guest feels safe and secure during their stay."
    ],

    'Location & Accessibility': [
        "The property is located just steps from public transport, making it easy to explore the city.",
        "Guests love the convenience of being near popular attractions.",
        "The location is central, making it easy to access everything.",
        "The property is within walking distance of shops and restaurants.",
        "Guests appreciate the easy access to major transportation hubs.",
        "The property is located in a quiet, yet accessible neighborhood.",
        "This home is perfectly situated for exploring the local area.",
        "The location makes it easy to travel to nearby landmarks.",
        "Guests love how close the property is to the subway station.",
        "The home is located in a safe and convenient area.",
        "The proximity to public transport makes getting around easy.",
        "The location is ideal for guests looking to explore the city.",
        "Guests appreciate being close to parks, shops, and cafes.",
        "The property is situated in a vibrant, accessible part of town.",
        "Guests love the short walk to local attractions and entertainment.",
        "The area is well-connected, making travel convenient.",
        "The central location makes this property a top choice for travelers.",
        "The neighborhood is safe and convenient for visitors.",
        "The property is ideally located for business travelers and tourists alike.",
        "Guests frequently compliment the home's easy accessibility to major sights."
    ],
    'Community & Neighborhood': [
        "The neighborhood is vibrant and full of local culture.",
        "Guests love the friendly and welcoming community.",
        "The area is safe and perfect for families and solo travelers alike.",
        "The neighborhood is filled with charming local shops and cafes.",
        "Guests appreciate the peaceful and quiet surroundings.",
        "The area offers a unique blend of history and modern living.",
        "The community vibe is friendly and welcoming.",
        "The local area is known for its cultural richness.",
        "Guests enjoy walking through the safe and quiet streets.",
        "The neighborhood is home to a diverse and lively community.",
        "The area is perfect for experiencing the local culture.",
        "Guests love the local markets and community events.",
        "The community is close-knit and always friendly to newcomers.",
        "The neighborhood feels safe and welcoming at all hours.",
        "The local culture and atmosphere are perfect for tourists.",
        "The area is known for its friendly, engaged community.",
        "T",
        "The peaceful, quiet streets are a guest favorite.",
        "The neighborhood is full of character, with unique local spots.",
        "Guests appreciate the sense of community and belonging in the area."
    ],

    'Interior Layout & Facilities': [
        "The layout of the home is spacious and well thought out.",
        "Every room is designed to maximize comfort and functionality.",
        "Guests love the modern, open-plan layout.",
        "The kitchen is fully equipped with high-end appliances.",
        "The bedrooms are well-sized and offer plenty of storage.",
        "The home features a beautiful and functional interior layout.",
        "The bathroom is modern, with sleek finishes and ample space.",
        "The open floor plan allows for a seamless flow between rooms.",
        "The living space is well-organized and easy to navigate.",
        "The layout ensures privacy and comfort for all guests.",
        "Guests appreciate the fully equipped kitchen and modern appliances.",
        "The interior design is both practical and aesthetically pleasing.",
        "The home is well-suited for both relaxation and socializing.",
        "The bathrooms are equipped with high-quality fixtures and plenty of space.",
        "The layout of the home is ideal for families or groups.",
        "The home features spacious living areas with plenty of natural light.",
        "The kitchen has everything needed for cooking and dining in.",
        "Guests love the functional layout and well-planned spaces.",
        "The bedrooms offer plenty of storage and are tastefully designed.",
        "The living areas are spacious, with plenty of seating and open space."
    ],

    'Technology & Equipment': [
        "The property is equipped with fast, reliable Wi-Fi.",
        "Guests love the modern technology available throughout the home.",
        "The smart home features make it easy to control lighting and temperature.",
        "The entertainment system includes a large TV with streaming services.",
        "The home offers multiple USB charging ports and smart plugs.",
        "Guests appreciate the high-speed internet, perfect for remote work.",
        "The smart lighting system can be controlled via voice commands.",
        "The home is equipped with modern appliances, including a dishwasher and washing machine.",
        "Guests enjoy the surround sound system and home theater setup.",
        "The property includes a fully automated security system for guest peace of mind.",
        "The home offers smart thermostats for easy temperature control.",
        "The property is equipped with high-tech entertainment systems.",
        "The smart TV is compatible with popular streaming platforms.",
        "Guests appreciate the convenience of voice-activated smart devices.",
        "The home features energy-efficient appliances throughout.",
        "The property offers high-speed internet, ideal for streaming and working online.",
        "The home has modern kitchen appliances, including a smart oven and fridge.",
        "The smart home system allows guests to easily control lights and temperature.",
        "Guests love the fast and reliable Wi-Fi, perfect for staying connected.",
        "The property is equipped with advanced technology for a modern living experience."
    ]
}
# 为主题生成嵌入
theme_embeddings = {theme: model.encode(sentences) for theme, sentences in theme_sentences.items()}


def initial_theme_match(review, df_dict):
    matched_themes = []
    for theme in df_dict['Sub Topic'].unique():
        keywords = df_dict[df_dict['Sub Topic'] == theme]['Keyword'].tolist()
        if any(keyword.lower() in review.lower() for keyword in keywords):
            matched_themes.append(theme)
    return matched_themes

# 为评论初步匹配候选主题
df_reviews['Initial Themes'] = df_reviews['Host Description'].apply(lambda review: initial_theme_match(review, df_dict))

# 计算每条评论的嵌入
df_reviews['Embedding'] = df_reviews['Host Description'].apply(lambda review: model.encode(review))


# 计算相似度并选择最相关的主题
def assign_final_theme(review_embedding, initial_themes, theme_embeddings):
    if not initial_themes:
        return None

    # 计算评论嵌入与每个候选主题的相似度
    similarities = {}
    for theme in initial_themes:
        theme_embedding = theme_embeddings[theme]
        sim = cosine_similarity([review_embedding], theme_embedding).mean()
        similarities[theme] = sim

    # 返回相似度最高的主题
    best_theme = max(similarities, key=similarities.get)
    return best_theme


# 为每条评论分配最终主题
df_reviews['Final Theme'] = df_reviews.apply(
    lambda row: assign_final_theme(row['Embedding'], row['Initial Themes'], theme_embeddings), axis=1)

# 使用 VADER 进行情感分析，将情感评分调整到 -5 到 5 之间
def sentiment_score(review):
    score = analyzer.polarity_scores(review)['compound']
    return score * 5  # 将得分从 -1 到 1 映射到 -5 到 5

# 为每条评论计算情感得分
df_reviews['Sentiment Score'] = df_reviews['Host Description'].apply(sentiment_score)

# 按主题计算情感得分的加权平均分和评论数量
theme_sentiment_summary = df_reviews.groupby('Final Theme').agg(
    Review_Count=('Final Theme', 'count'),               # 统计每个主题的评论数量
    Average_Sentiment_Score=('Sentiment Score', 'mean')  # 计算每个主题的情感加权平均分
).reset_index()

# 输出结果到 CSV 文件
theme_sentiment_summary.to_csv('host_theme_sentiment_summary.csv', index=False)


# 查看结果
print(theme_sentiment_summary)
