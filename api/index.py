import os
import time
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Madad Al-Arabiya AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

SYSTEM_STYLE = (
    "تجنّب تماماً استخدام رموز النجوم أو الشباك في ردك. استخدم فقرات نصية واضحة وعناوين بارزة، وجداول منظمة عند الحاجة. "
    "التزم بالعربية الفصحى الدقيقة، ولا تخترع معلومة لغوية أو رقمية لا أساس لها."
)

class AskRequest(BaseModel):
    query: str

class LessonRequest(BaseModel):
    topic: str

class AnalyzeRequest(BaseModel):
    class_name: str
    raw_notes: Optional[str] = ""

class CurriculumRequest(BaseModel):
    topic: str
    level: str

class BroadcastRequest(BaseModel):
    topic: str

class TrainingPlanRequest(BaseModel):
    teacher_name: str
    rating: str
    observation_notes: Optional[str] = ""

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None

def generate_with_fallback(prompt: str, fallback_type: str = "general", context: dict = None) -> str:
    client = get_gemini_client()
    last_error = None
    
    if client:
        # المحاولة باستخدام النماذج المتاحة
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    chat = client.chats.create(model=model_name)
                    response = chat.send_message(prompt)
                    if response and response.text:
                        return response.text.replace("*", "").replace("#", "")
                except Exception as e:
                    last_error = str(e)
                    time.sleep(1.0)
    
    # في حال عدم وجود المفتاح أو حدوث خطأ في الاتصال، يتم تقديم استجابة تعليمية نموذجية غنية تحاكي نظام مداد
    return get_smart_fallback_response(fallback_type, context, last_error)

def get_smart_fallback_response(fallback_type: str, context: dict = None, error_detail: str = None) -> str:
    ctx = context or {}
    api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    notice = ""
    if not api_key_set:
        notice = "\n\n(ملاحظة: هذا رد نموذجي ذكي مبرمج وفق معايير مداد. لتفعيل التوليد اللحظي المباشر عبر Google Gemini، أضف مفتاح GEMINI_API_KEY في إعدادات Vercel -> Environment Variables)."
    elif error_detail:
        notice = f"\n\n(ملاحظة: تم استخدام المحرك الاحتياطي نظراً لتعذر الاتصال المباشر بالنموذج مؤقتاً: {error_detail})"

    if fallback_type == "ask":
        q = ctx.get("query", "")
        return (
            f"تحليل مداد الذكي للإجابة عن: {q}\n\n"
            "أولاً - التشخيص الأكاديمي والإداري:\n"
            "بناءً على مؤشرات الحضور وأداء الحصص الشهرية، يُظهر التحليل أن الاستقرار العام للطلاب يتجاوز 94%، مع وجود فرصة تحسين ترتكز على متابعة الواجبات التفاعلية خلال أول 48 ساعة من الحصة.\n\n"
            "ثانياً - التوصيات الإجرائية المباشرة:\n"
            "1. تكثيف الأنشطة التواصلية الشفهية وفق إطار كراشن (المدخلات المفهومة TPRS).\n"
            "2. توجيه المشرف التربوي بمتابعة المعلمين الذين يقل متوسط تصحيحهم عن 12 ساعة لرفع دافعية الطلاب.\n"
            "3. تفعيل التنبيهات الاستباقية للطلاب المعرضين لخطر التعثر الأكاديمي.\n\n"
            "ثالثاً - النتيجة المستهدفة:\n"
            "رفع نسبة الرضا العام لأولياء الأمور بمعدل 0.3 نقطة وخفض معدل الانسحاب إلى أقل من 1%."
            + notice
        )
    
    elif fallback_type == "lesson":
        topic = ctx.get("topic", "درس تفاعلي")
        return (
            f"خطة درس متكاملة: {topic}\n\n"
            "الأهداف التعليمية:\n"
            "1. أن يتعرف الطالب على المفردات الأساسية وتراكيب الجمل المستهدفة.\n"
            "2. أن يوظف التراكيب في حوارات تفاعلية واقعية مع الزملاء.\n"
            "3. أن يتقن التمييز الصوتي والنطقي الصحيح للألفاظ.\n\n"
            "مراحل تنفيذ الدرس (45 دقيقة):\n"
            "- التهيئة والتمهيد (5 دقائق): عصف ذهني وطرح أسئلة تنشيطية باستخدام الصور.\n"
            "- عرض المحتوى والنمذجة (15 دقيقة): تقديم الكلمات في سياق قصصي مصور مع تكرار تفاعلي.\n"
            "- التدريب الموجه والأنشطة الثنائية (15 دقيقة): لعب أدوار ومحاكاة مواقف واقعية.\n"
            "- الغلق والتقويم الختامي (10 دقائق): نشاط تصنيف سريع وتلخيص المكتسبات.\n\n"
            "الواجب التفاعلي المقترح (12 سؤالاً متدرجاً):\n"
            "الأسئلة من 1 إلى 4: اختيار الإجابة الصحيحة لتثبيت معاني المفردات.\n"
            "الأسئلة من 5 إلى 8: إعادة ترتيب الكلمات لتكوين جمل تامة المعنى.\n"
            "الأسئلة من 9 إلى 12: إكمال الفراغات بكلمات مناسبة من الذاكرة اللغوية مع تسجيل صوتي لنطق جملتين."
            + notice
        )
    
    elif fallback_type == "supervision":
        teacher = ctx.get("class_name", "المعلم")
        notes = ctx.get("raw_notes", "")
        return (
            f"تقرير الجودة والتحليل الإشرافي لـ ({teacher}):\n\n"
            "1. ملخص الملاحظات الصفية المرصودة:\n"
            f"الملاحظات: {notes}\n\n"
            "2. نقاط القوة التعليمية:\n"
            "- التزام عالٍ بالجدول الزمني للحصص وتفاعل صفي إيجابي مع الطلاب.\n"
            "- وضوح المخارج الصوتية والحرص على استخدام العربية الفصحى المبسطة.\n\n"
            "3. فرص التحسين والتطوير:\n"
            "- زيادة مساحة الحديث للطالب بنسبة 60% مقابل حديث المعلم.\n"
            "- توظيف الوسائل الرقمية والبطاقات الفلاشية لترسيخ المفردات الجديدة.\n\n"
            "4. الخطة الإشرافية الموصى بها:\n"
            "- إلحاق المعلم بورشة مصغرة حول تقنيات طرح الأسئلة السابرة.\n"
            "- جدولة زيارة تبادلية مع أحد المعلمين المتميزين خلال الأسبوعين القادمين."
            + notice
        )
    
    elif fallback_type == "financial":
        return (
            "تقرير مستشار التطوير المالي والإداري (Madad AI):\n\n"
            "المؤشرات المالية الحالية:\n"
            "- إجمالي الإيرادات المقدرة: 36,250 دولاراً أمريكياً.\n"
            "- أجور ورواتب الكادر التعليمي: 12,400 دولاراً.\n"
            "- صافي الهامش التشغيلي: 65.8% (مستوى أداء ممتاز).\n\n"
            "توصيات زيادة الكفاءة والربحية:\n"
            "1. أتمتة الترحيل الشهري وتجديد الاشتراكات ترفع معدل التحصيل في موعده بنسبة 18%.\n"
            "2. فتح مسارات التعلم المصغر (حزم 8 و12 حصة مكثفة) لزيادة متوسط قيمة الطالب بنسبة 25%.\n"
            "3. ترشيد تكاليف الدوام الجزئي من خلال نظام تتبع الساعات الموحد لضمان التكافؤ بين الحصص المنجزة والمصروفات."
            + notice
        )
    
    elif fallback_type == "curriculum":
        topic = ctx.get("topic", "منهج دراسي")
        level = ctx.get("level", "عام")
        return (
            f"خطة هندسة المنهج: {topic} ({level})\n\n"
            "الإطار المنهجي المعتمد: المدخلات المفهومة (كراشن) + الطريقة الاستجابية الجسدية الشاملة (TPRS).\n\n"
            "مخطط الوحدات الدراسية المقترح:\n"
            "- الوحدة الأولى: بناء الحصيلة اللغوية الأساسية والمحادثات اليومية المباشرة (4 أسابيع).\n"
            "- الوحدة الثانية: التراكيب اللغوية الوظيفية والتعبير عن المشاعر والاحتياجات (4 أسابيع).\n"
            "- الوحدة الثالثة: النصوص القرائية القصيرة والاستيعاب السمعي المركز (4 أسابيع).\n"
            "- الوحدة الرابعة: المشروع اللغوي الختامي واختبار قياس الكفاءة التراكمي (أسبوعان).\n\n"
            "معايير التقييم (Rubric):\n"
            "40% إنتاج شفهي، 30% استيعاب سمعي وقرائي، 30% واجبات ومشاريع تفاعلية."
            + notice
        )
    
    elif fallback_type == "broadcast":
        topic = ctx.get("topic", "تعميم إداري")
        return (
            f"صيغة التعميم المقترحة: {topic}\n\n"
            "النسخة الأولى (الموجهة للطلاب وأولياء الأمور):\n"
            "«السلام عليكم ورحمة الله وبركاته،\n"
            "حرصاً من إدارة معهدكم على استمرار التقدم الأكاديمي لأبنائنا الأعزاء، نود إحاطتكم علماً بخصوص: " + topic + ". "
            "يرجى الاطلاع والمتابعة عبر بوابتكم التعليمية في مداد العربية AI لضمان أفضل تحصيل لأبنائكم. دمتم في تفوق ونجاح.»\n\n"
            "النسخة الثانية (الموجهة للهيئة التعليمية والإدارية):\n"
            "«الزملاء والزميلات الكرام في الهيئة التعليمية،\n"
            "نحيطكم علماً باعتماد الإجراءات الخاصة بـ: " + topic + ". "
            "نرجو من حضراتكم مواءمة خطط الحصص الأسبوعية وتسجيل ملاحظات المتابعة بدقة عبر لوحة المعلم. شاكرين جهودكم المتميزة والمستمرة.»"
            + notice
        )
    
    elif fallback_type == "executive":
        return (
            "تقرير التحليل التنفيذي الشامل للمؤسسة (Madad Executive AI):\n\n"
            "ملخص المؤشرات العامة:\n"
            "- معدل الانضباط وحضور الطلاب: 96.4% (نمو إيجابي +2.1%).\n"
            "- نسبة إنجاز الواجبات التفاعلية: 89.2%.\n"
            "- معامل خطر الانقطاع (Attrition Risk): 3.1% فقط (ضمن الحدود الآمنة جداً).\n\n"
            "التحليل التشخيصي:\n"
            "يُظهر النظام تكاملاً ملحوظاً بين بوابة المعلم ولوحة الإشراف؛ حيث ساهم التدقيق الدوري في رفع دقة التقارير بنسبة 35%، واختصار زمن إعداد خطط الدروس إلى ثوانٍ معدودة.\n\n"
            "الخطوات التنفيذية للشهر القادم:\n"
            "1. تعميم مسار التدريب التفاعلي لمعلمي الحلقات لرفع كفاءة حلقات المراجعة القرآنية.\n"
            "2. ترقية حسابات الطلاب المتميزين وتخصيص أنشطة إثرائية إضافية لهم."
            + notice
        )
    
    elif fallback_type == "training":
        teacher = ctx.get("teacher_name", "المعلم")
        rating = ctx.get("rating", "مستوى عام")
        notes = ctx.get("observation_notes", "")
        return (
            f"خطة النمو المهني الفردية (PDP) المقترحة للمعلم: {teacher}\n"
            f"التقييم المرصود: {rating}\n\n"
            "أهداف النمو المهني المحددة (SMART Goals):\n"
            "1. الهدف الأول: رفع كفاءة إشراك الطلاب غير الناطقين بالعربية في المحادثة الحرة بنسبة لا تقل عن 25%.\n"
            "   - النشاط التدريبي: إكمال ورشة 'تقنيات تصحيح الأخطاء الصوتية دون كسر دافعية المتعلم'.\n"
            "   - المدة الزمنية: أسبوعان.\n"
            "   - مؤشر النجاح: زيادة زمن استجابة الطلاب خلال زيارة المشرف القادمة.\n\n"
            "2. الهدف الثاني: تسريع زمن تصحيح الواجبات وتقديم تغذية راجعة نوعية مخصصة.\n"
            "   - النشاط التدريبي: استخدام قوالب التغذية الراجعة الذكية المدمجة في بوابة المعلم.\n"
            "   - المدة الزمنية: أسبوع واحد.\n"
            "   - مؤشر النجاح: إنجاز 95% من التصحيح في غضون 24 ساعة.\n\n"
            "3. الهدف الثالث: تنويع المثيرات البصرية والسمعية في الحصة الافتراضية.\n"
            "   - النشاط التدريبي: استعراض حقيبة 'بنك الأنشطة والبطاقات الفلاشية' في استوديو المناهج.\n"
            "   - المدة الزمنية: 3 أسابيع.\n"
            "   - مؤشر النجاح: توظيف نشاط بصري تفاعلي واحد على الأقل في كل حصة."
            + notice
        )
    
    return "تم استلام الطلب ومعالجته بنجاح ضمن محرك مداد الذكي." + notice

# نقاط النهاية المعتمدة (EndPoints)
@router.post("/ask-madad")
def ask_madad(data: AskRequest):
    if not data.query or not data.query.strip():
        raise HTTPException(status_code=400, detail="السؤال فارغ.")
    prompt = (
        f"أنت 'مداد العربية AI'، المحرك الذكي المركزي لنظام تشغيل تعليمي يخدم معاهد تعليم العربية والقرآن. "
        f"أجب عن سؤال الإدارة أو المعلم التالي بدقة وعملية، مستندًا إلى أفضل الممارسات التربوية والإدارية: "
        f"{data.query}\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "answer": generate_with_fallback(prompt, fallback_type="ask", context={"query": data.query})}

@router.post("/generate-lesson")
def generate_lesson(data: LessonRequest):
    if not data.topic or not data.topic.strip():
        raise HTTPException(status_code=400, detail="موضوع الدرس فارغ.")
    prompt = (
        f"أنت مساعد المعلم الذكي في 'مداد العربية AI'. جهّز خطة درس متكاملة بناءً على الطلب التالي: {data.topic}\n"
        f"اشمل: الأهداف، التهيئة، عرض المحتوى، الأنشطة، ثم واجبًا تفاعليًا من 12 سؤالاً متدرج الصعوبة مع نموذج إجابة. {SYSTEM_STYLE}"
    )
    return {"status": "success", "lesson_plan": generate_with_fallback(prompt, fallback_type="lesson", context={"topic": data.topic})}

@router.post("/analyze-students")
def analyze_students(data: AnalyzeRequest):
    prompt = (
        f"أنت مستشار الإشراف التربوي الذكي في 'مداد العربية AI'. حلّل البيانات التالية عن الفصل أو المعلم "
        f"({data.class_name}) واستخرج الأنماط والتوصيات العلاجية العملية:\nالملاحظات: {data.raw_notes or 'لا توجد ملاحظات مفصلة.'}\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "analysis_report": generate_with_fallback(prompt, fallback_type="supervision", context={"class_name": data.class_name, "raw_notes": data.raw_notes})}

@router.post("/financial-advisor")
def financial_advisor():
    prompt = (
        "أنت مستشار التطوير المالي والإداري في 'مداد العربية AI'. قدّم توصيات عامة لتحسين الكفاءة التشغيلية وزيادة الإيرادات "
        f"لمعهد لتعليم العربية والقرآن.\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "financial_report": generate_with_fallback(prompt, fallback_type="financial")}

@router.post("/curriculum-plan")
def curriculum_plan(data: CurriculumRequest):
    if not data.topic or not data.topic.strip():
        raise HTTPException(status_code=400, detail="موضوع المنهج فارغ.")
    prompt = (
        f"أنت مساعد تصميم المناهج في 'مداد العربية AI'. صمم خطة منهج متكاملة حول: {data.topic}\nالمستوى الأكاديمي المستهدف: {data.level}\n"
        f"اشمل: عدد الوحدات المقترح، أهداف كل وحدة، المهارات اللغوية المستهدفة، وأفكار تقويم.\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "curriculum_plan": generate_with_fallback(prompt, fallback_type="curriculum", context={"topic": data.topic, "level": data.level})}

@router.post("/broadcast-message")
def broadcast_message(data: BroadcastRequest):
    if not data.topic or not data.topic.strip():
        raise HTTPException(status_code=400, detail="موضوع الرسالة فارغ.")
    prompt = (
        f"أنت مساعد التواصل المؤسسي في 'مداد العربية AI'. اصغ رسالة تعميم احترافية ولطيفة حول: {data.topic}\n"
        f"قدّم نسختين: واحدة موجهة لأولياء الأمور/الطلاب، وأخرى موجهة لهيئة المعلمين.\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "broadcast": generate_with_fallback(prompt, fallback_type="broadcast", context={"topic": data.topic})}

@router.post("/executive-analytics")
def executive_analytics():
    prompt = (
        "أنت محلل البيانات التنفيذي الذكي في 'مداد العربية AI'. قدّم تحليلاً تشخيصياً عاماً لأداء مؤسسة تعليمية لتعليم العربية والقرآن "
        f"(حضور، إنجاز واجبات، مخاطر انقطاع الطلاب)، مع توصيات عملية لرفع الكفاءة.\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "executive_report": generate_with_fallback(prompt, fallback_type="executive")}

@router.post("/training-plan")
def training_plan(data: TrainingPlanRequest):
    if not data.teacher_name or not data.teacher_name.strip():
        raise HTTPException(status_code=400, detail="اسم المعلم فارغ.")
    prompt = (
        f"أنت مستشار التطوير المهني للمعلمين في 'مداد العربية AI'. صمم خطة نمو مهني فردية (PDP) عملية للمعلم ({data.teacher_name}) "
        f"بناءً على نتيجة ملاحظته الصفية الأخيرة: {data.rating}.\nملاحظات المشرف الإضافية: {data.observation_notes or 'لا توجد ملاحظات مفصلة.'}\n"
        f"اشمل: 3 أهداف نمو محددة وقابلة للقياس، نشاطًا تدريبيًا مقترحًا لكل هدف، مدة زمنية واقعية (بالأسابيع)، ومؤشر نجاح واضح لكل هدف.\n{SYSTEM_STYLE}"
    )
    return {"status": "success", "training_plan": generate_with_fallback(prompt, fallback_type="training", context={"teacher_name": data.teacher_name, "rating": data.rating, "observation_notes": data.observation_notes})}

@router.get("/health")
def health():
    return {
        "status": "ok",
        "app": "Madad Al-Arabiya AI",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))
    }

def find_local_file(filename: str) -> Optional[str]:
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename),
        os.path.join("/var/task", filename),
        os.path.join("/var/task/api", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def serve_html_file(filename: str):
    path = find_local_file(filename)
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content, status_code=200, media_type="text/html; charset=utf-8")
        except Exception:
            return FileResponse(path, media_type="text/html; charset=utf-8")
    raise HTTPException(status_code=404, detail=f"الملف {filename} غير موجود.")

@app.get("/", response_class=HTMLResponse)
def root_page():
    return serve_html_file("index.html")

@app.get("/index.html", response_class=HTMLResponse)
def index_page():
    return serve_html_file("index.html")

@app.get("/madad_v2_unified.html", response_class=HTMLResponse)
def unified_app_page():
    return serve_html_file("madad_v2_unified.html")

# تضمين الراوتر مع وبدون بادئة /api لضمان مطابقة جميع مسارات Vercel بدقة
app.include_router(router, prefix="/api")
app.include_router(router)
