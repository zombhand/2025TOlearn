# MyBatisPlus

mp只支持单表查询，你用到多个mapper还叫单表吗

目标基于MyBatisPlus 完成标准DTO业务开发

MyBatisPlus（简称MP）是基于MyBatis框架基础上开发的增强型工具，旨在简化开发、提高效率

```java
@Mapper
public interface UserDao extends BaseMapper<User> {}
```

MyBatisPlus特性

- 无侵入：只做增强不做改变，不会对现有工程产生影响
- 强大的 CRUD 操作：内存通用 Mapper，少量配置即可完成单表CRUD 操作
- 支持 Lambda：编写查询条件无需担心字段写错
- 支持主键自动生成
- 内置分页插件
- 类名驼峰转下划线作为表名
- 名为id的字段作为主键
- 变量名驼峰转下划线作为表的字段名

常见注解

- @TableName: 用来指定表名
- @Tableld：用来指定表中的主键字段信息
- @TableField：用来指定表中的普通字段信息

| 功能       | 自定义接口                             | MP接口                                       |
| ---------- | -------------------------------------- | -------------------------------------------- |
| 新增       | boolean save(T t)                      | int insert(T t)                              |
| 删除       | boolean delete(int id)                 | int deleteById(Serializable id)              |
| 修改       | boolean update(T t)                    | int updateById(T t)                          |
| 根据id查询 | T getById(int id)                      | T selectById(Serializable id)                |
| 查询全部   | List<T> getAll()                       | List<T> selectList()                         |
| 分页查询   | PageInfo<T> getAll(int page, int size) | IPage<T> selectPage(IPage<T> page)           |
| 按条件查询 | List<T> getAll(Condition condition)    | IPage<T> selectPage(Wrapper<T> queryWrapper) |

## 使用的基本流程

1. 引入起步依赖
2. 自定义Mapper基础BaseMapper
3. 在实体类上添加注解声明 表信息

## 条件构造器

MyBatisPlus 支持各种复杂的where条件，可以满足日常开发的需求

![image-20250102155259617](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250102155259617.png)



### 分页查询实现

tap2

```java
@Test
void testGetByPage(){
    IPage page = new Page(1, 2);
    userDao.selectPage(page, null);
    System.out.println("当前页码值："+page.getCurrent());
    System.out.println("每页显示数："+page.getSize());
    System.out.println("一共多少页："+page.getPages());
    System.out.println("一共多少条数据："+page.getTotal());
    System.out.println("数据："+page.getRecords());
}
```

tap1

需要为其配置一个拦截器

```java
@Configuration
public class MpConfig {
    
    @Bean
    public MybatisPlusInterceptr mpInterceptor(){
        // 1.定义Mp拦截器
        MybatisPluseInterceptor mpInterceptor = new MybatisPluseInterceptor();
        // 2.添加具体的拦截器
        mpInterceptor.addInnerInterceptor(new PaginationInnerInterceptor());
        return mpInterceptor;
    }
}
```

```yaml
# 开启mp 的日志（输出到控制台）
mybatis-plus:
	configuration:
		log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
```

### DQL编程控制

**条件查询**

banner（是配置文件的icon是否显示的控制关键字）

```java
@Test
void testGetAll() {
    // 方式一：按条件查询
    QueryWrapper qw = new QueryWrapper();
    qw.lt("age", 18);
    List<User> userList = userDao.selectList(qw);
    System.out.println(userList);
    
    // 方式二：lambda 格式按条件查询
    QueryWrapper<User> qw = new QueryWrapper<User>();
    qw.lambda().lt(User::getAge, 18);
    List<User> userList = userDao.selectList(qw);
    System.out.println(userList);
    
    // 方式三：lambda格式按条件查询
    LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
    lqw.lt(User::getAge, 10);
    List<User> userList = userDao.selectList(lqw);
    System.out.println(userList);
    
    // 多条件
    LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
    // 10到30岁之间
    //lqw.lt(User::getAge, 30);
    //lqw.gt(User::getAge, 10);
    // 链式编程
    lqw.lt(User::getAge, 30).gt(User::getAge, 10);
    // 小于10岁或者大于30岁
    lqw.lt(User::getAge, 10).or().gt(User::getAge, 10);
    List<User> userList = userDao.selectList(lqw);
    System.out.println(userList);
    
    // 条件查询——null值处理
    // 先判定第一个参数是否为 true，如果为true连接当前条件
    UserQuery uq = new UserQuery();
    uq.setAge2(30);
    uq.setAge(10);
    LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
    lqw.lt(null != uq.getAge(), User::getAge, uq.getAge())
       .rt(null != uq.getAge2(), User::getAge, uq.getAge2());
    List<User> userList = userDao.selectList(lqw);
    System.out.println(userList);
    
    //查询投影
    //LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
   	//lqw.select(User::getId,User::getName,User::getAge); // 只查看此3类数据
    QueryWrapper<User> lqw = new QueryWrapper<User>();
    lqw.selcet("id","name","age","tel");
    List<User> userList = userDao.selectList(lqw);
    System.out.println(userList);
    
    QueryWrapper<User> lqw = new QueryWrapper<User>();
    lqw.selcet("count(*) as count");
    List<Map<String, Object>> userList = userDao.selectMaps(lqw);
    System.out.println(userList);
    
     QueryWrapper<User> lqw = new QueryWrapper<User>();
    lqw.selcet("count(*) as count, tel");
    lqw.groupBy("tel");
    List<Map<String, Object>> userList = userDao.selectMaps(lqw);
    System.out.println(userList);
}
```

**查询条件**

- 范围匹配（> 、 = 、between）
- 模糊匹配（like）
- 空判定（null）
- 包含性匹配（in）
- 分组（group）
- 排序（order）
- ....

```java
@Test
void testGetAll() {
    // 条件查询
    LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
    // 等同于 =
    lqw.eq(User::getName, "Jerry").eq(User::getPassword, "jerry");
    List<User> userList = userDao.selectList(lqw); 
    // 如果只有一个
    User loginUser = userDao.selectOne(lqw);
    System.out.println(userList);
    
    LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
    // 范围查询 lt le gt ge == < <= > >=
    lqw.between(User::getAge, 10, 30);
    List<User> userList = userDao.selectList(lqw);
    System.out.println(userList);
    
    LambdaQueryWrapper<User> lqw = new LambdaQueryWrapper<User>();
    // 模糊匹配 like
    lqw.like(User::getName, "J"); // == "%J%"
    lqw.likeLeft(User::getName, "J"); // == "%J"
    lqw.likeRight(User::getName, "J"); // == "J%"
    List<User> userList = userDao.selectList(lqw);
    System.out.println(userList);
}
```

### 字段映射与表名映射

- 名称：@TableField

- 类型：**属性注解**

- 位置：模型类属性定义上方

- 作用：设置当前属性对应的数据库表中的字段关系

- 范例：

- ```java
  public class User{
      @TableField(value = "pwd")
      private String password;
  }
  ```

- 相关属性

  - value（默认）：设置数据库表字段名称

- 问题二：编码中添加了数据库中未定义的属性

- ```mysql
  CREATE TABLE `user` (
  	`id` bigint(20) NOT NULL AUTO_INCREMENT.
      `name` varchar(32),
      `pwd` varchar(32),
      `age` int(3),
      `tel` varchar(32),
      PRIMARY KEY (`id`)
  )
  ```

  ```java
  public class User {
      private Long id;
      private String name;
      @TableField(value="pwd")
      private String password;
      private Integer age;
      private String tel;
      private Integer online;
  }
  ```

- 名称：@TableField

- 类型：属性注解

- 位置：模型类属性定义上方

- 作用：设置当前属性对应的数据库表中的字段关系

- 范例：

- ```java
  public class User {
      @TableField(exist = false)
      private Integer online;
  }
  ```

- 相关属性

  - value：设置数据库表字段名称
  - **exist：设置属性在数据库表字段中是否存在，默认为true。此属性无法与value合并使用**

- 问题三：采用默认查询开放了更多的字段查看权限

- 名称：@TableField

- 类型：属性注解

- 位置：模型类属性定义上方

- 作用：设置当前属性对应的数据库表中的字段关系

- 范例：

- ```java
  public class User{
      @TableField(value = "pwd",select = false)
      private String password;
  }
  ```

- 相关属性

  - select:设置属性是否参与查询，此属性与select()映射配置不冲突

- 问题四：表名与编码开发设计不同步

- ```java
  @TableName("tbl_user")
  public class User {
      private Long id;
      private String name;
      @TableField(value="pwd",select = false)
      private String password;
      private Integer age;
      private String tel;
      @TableField(exist = false)
      private Integer online;
  }
  ```

- 名称：@TableName

- 类型：类注解

- 位置：模型类定义上方

- 作用：设置当前类对应与数据库表关系

- 相关属性

  - value：设置数据库表名称
  
- 问题五：成员变量名以is开头，且是布尔值

常见场景：

- 成员变量名与数据库字段名不一致
- 成员变量名以is开头，且是布尔值
- 成员变量名与数据库关键字冲突
- 成员变量不是数据库字段

```java
@TableName("tb_user")
public class User {
    @TableId(value="id", type = IdType.AUTO)
    private Long id;
    @TableField("username")
    private String name;
    @TableField("is_married")
    private Boolean isMarried;
    @TableField("`order`")
    private Integer order;
    @TableField(exist = false)
    private String address;
}
```



### **id 生成策略控制**

- 不同的表应用不同的id生成策略

  - 日志：自增（1,2,3,4，....）
  - 购物订单：特殊规则（FQ23948AK3843）
  - 外卖单：关联地区日期等信息（10 04 20200314 34 91）
  - 关系表：可省略id
  - ....

- 名称：@TableId

- 类型：属性注解

- 位置：模型类中用于表示主键的属性定义上方

- 作用：设置当前类中主键属性的生成策略

- 范例：

- ```java
  public class User {
      @TableId(type = IdType.AUTO)
      private Long id;
  }
  ```

- 相关属性

  - value：设置数据库主键名称
  - type：设置主键属性的生成策略，值参照 IdType枚举值

- AUTO(0)：使用数据库id自增策略控制id生成

- NONE(1)：不设置id生成策略

- INPUT(2)：用户手工输入id

- ASSIGN_ID(3)：雪花算法生成id（可兼容数值型与字符串型）

- ASSIGN_UUID(4)：以UUID生成算法作为id生成策略

`0|00100110111011010101100001101010011000110|10000|10001|000000000010`

占位符：0	时间戳(41) 	机器码(5+5)	序列号(12)

最大程度上确保id不重复防止多线程下的数据

也可以在配置中 全局配置这样的规则设定

```yaml
mybatis-plus:
	global-config:
		db-config:
			id-type: auto
			table-prefix: tbl_
```

### 多记录操作

- 按照主键删除多条记录

- ```java
  List<Long> ids = Arrays.asList(new Long[]{2,3});
  userDao.deleteBatchIds(ids);
  ```

- 根据主键查询多条记录

- ```java
  List<Long> ids = Arrays.asList(new Long[]{2,3});
  List<User> userList = userDao.selectBatchIds(ids);
  ```

### 逻辑删除

- 删除操作业务问题：业务数据从数据库中丢弃
- 逻辑删除：为数据设置是否可用状态字段，删除时设置状态字段为不可用状态，数据保留在数据库中

步骤

1. 数据库表中添加逻辑删除标记字段 int

2. 实体类中添加对应字段，并设定当前字段为逻辑删除标记字段

   ```java
   public class {
      private Long id;
      @TableLogic
      private Integer deleted;
   }
   ```

3. 配置逻辑删除字面值

   ```yaml
   mybatis-plus:
   	global-config:
   		db-config:
   			logic-delete-field: deleted
   			logic-not-delete-value: 0
   			logic-delete-value: 1
   ```

   执行的具体SQL语句：`UPDATE tbl_user SET deleted = 1 WHERE id = ? AND deleted = 0`

## my配置

```yml
mybatis-plus:
	type-aliases-package: com.itheima.mp.domain.po #别名扫描包
	mapper-locations: "classpath*:/mapper/**/*.xml" #Mapper.xml文件地址，默认值
	configuration:
		map-underscore-to-camel-case: true #是否开启下划线和驼峰的映射
		cache-enabled: false #是否开启二级缓存
	global-config:
		db-config:
			id-type: assign_id # id为雪花算法生成
			update-strategy: not_null # 更新策略：只更新非空字段
```

## 自定义SQL

利用MyBatisPlus的Wrapper来构建复杂的where条件，然后自己定义SQL语句中剩下的部分。

1. 基于 Wrapper构建where条件

   ```java
   List<Long> ids = List.of(1L, 2L, 4L);
   int amount = 200;
   // 1.构建条件
   LambdaQueryWrapper<User> wrapper = new LambdaQUeryWrapper<User>().in(User::getId, ids);
   // 2.自定义SQL方法调用
   userMapper.updateBalanceByIds(wrapper, amount);
   ```

2. 在mapper方法参数中用Param注解声明wrapper变量名称，必须是ew

   ```java
   void updateBalanceByIds(@Param("ew") LambdaQueryWrapper<User> wrapper, @Param("amount") int amount);
   ```

3. 自定义SQL，并使用Wrapper条件

   ```xml
   <update id="updateBalanceByIds">
       UPDATE tb_user SET balance = banlance - #{amount} ${ew
       .customSqlSegment}
   </update>    
   ```

## 核心功能-IService接口



![image-20250102164618335](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250102164618335.png)

![image-20250102164602862](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250102164602862.png)

使用流程：

1. 定义接口Service去继承IService
2. 写一个接口的实现类
3. 在实现类中继承mp给的实现类

ServiceImpl 是mp给的实现类

### Lambda使用

lambda查询

- list()查询一个集合
- one()查询单个案例
- count()总数
- 分页page

lambda更新

### IService 批量新增

需求：批量插入10万条

- 普通for循环插入

- IService 的批量插入
- 开启rewriteBatchedStatements=true参数

批处理方案:

- 普通for循环逐条插入速度极差，不推荐
- MP的批量新增，基于预编译的批处理，性能不错
- 配置jdbc参数，开rewriteBatchedStatements，性能最好

## 静态工具

`&rewriteBatchedStatemments=true`

![image-20250112180044374](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250112180044374.png)

## 逻辑删除

**逻辑删除**就是基于代码逻辑模拟删除效果，但并不会真正删除数据。

- 在表中添加一个字段标记数据是否被删除
- 当删除数据时把标记置为1
- 查询时只查询标记为0的数据

**MyBatisPlus**提供了逻辑删除功能，无需改变方法调用的方式，而是在底层帮我们自动修改CRUD的语句。我们要做的就是在application.yaml文件中配置逻辑删除的字段名称和值即可：

```yaml
mybatis-plus:
	global-config:
		db-config:
			logic-delete-field: flag # 全局逻辑删除的实体字段名，字段类型可以是 boolean、integer
			logic-delete-value: 1 # 逻辑已删除值（默认为1）
			logic-not-delete-value: 0 # 逻辑未删除值（默认为0）
```

逻辑删除本身也有问题:

- 会导致数据库表垃圾数量越来越多，影响查询效率
- SQL中全都需要对逻辑删除字段做判断，影响查询效率

因此，此方法不太推荐进行使用，如果数据不能删除，可以采用把数据迁移到其他表的办法。

## 枚举处理器

![image-20250112205750323](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250112205750323.png)

如何实现PO类中的枚举类型变量与数据库字段的转换？

1. 给枚举中的与数据库对应value值添加@EnumValue注解

2. 在application.yml中配置全局枚举处理器：

   ```yaml
   mybatis-plus:
   	configuration:
   		default-enum-type-handler: com.baomidou.mybatisplus.core.handlers.MybatisEnumTypeHandler
   ```

## JSON处理器

![image-20250112212920848](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250112212920848.png)



## 内置拦截器

![image-20250114095443578](C:\Users\29326\AppData\Roaming\Typora\typora-user-images\image-20250114095443578.png)

### 分页插件

1. 实现分页的配置

   ```java
   @Configuration
   public class MyBatisConfig {
   	
       @Bean
       public MybatisPlusInterceptor mybatisPlusInterceptor() {
           MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
           // 1.添加分页插件
           PaginationInnerIntercepotr paginationInnerInterceptor = new PaginationInnerInterceptor(DbType.MYSQL);
           paginationInnerInterceptor.setMaxLimit(1000L);
           
           // 2.添加分页插件
           interceptor.addInnerInterceptor(paginationInnerInterceptor);
           return interceport;
       }
   }
   
   ```

2. 调用分页功能

   ```java
   int pageNo = 1, pageSize = 2;
   
   // 1.准备分页条件
   // 1.1 分页条件
   Page<User> page = Page.of(pageNo, PageSize);
   // 1.2 排序条件
   page.addOrder(new OrderItem("balance", true));
   page.addOrder(new OrderItem("id", true));
   
   // 2.分页查询
   Page<User> p = userService.page(page);
   
   // 3.解析
   long total = p.getTotal();
   long pages = p.getPages();
   List<User> users = p.getRecords();
   ```

   
